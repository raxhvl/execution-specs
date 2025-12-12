"""
Tests for EIP-7928 Block Access Lists with single-opcode success and OOG
scenarios.

Block access lists (BAL) are generated via a client's state tracing journal.
Residual journal entries may persist when opcodes run out of gas, resulting
in a bloated BAL payload.

Issues identified in:
https://github.com/paradigmxyz/reth/issues/17765
https://github.com/bluealloy/revm/pull/2903

These tests ensure out-of-gas operations are not recorded in BAL,
preventing consensus issues.
"""

from enum import Enum
from typing import Callable, Dict

import pytest
from rich.console import Console
from rich.table import Table

from execution_testing import (
    AccessList,
    Account,
    Address,
    Alloc,
    BalAccountExpectation,
    BalBalanceChange,
    BalNonceChange,
    BalStorageChange,
    BalStorageSlot,
    Block,
    BlockAccessListExpectation,
    BlockchainTestFiller,
    Bytecode,
    Fork,
    Initcode,
    Op,
    Transaction,
    compute_create_address,
)

from .spec import ref_spec_7928

REFERENCE_SPEC_GIT_PATH = ref_spec_7928.git_path
REFERENCE_SPEC_VERSION = ref_spec_7928.version


pytestmark = pytest.mark.valid_from("Amsterdam")


# Global list to collect gas data for aggregate reporting
_GAS_TRACKING_DATA: list = []

# Global list for test_bal_delegatecall_no_delegation_and_oog_before_target_access
_TEST_BAL_DELEGATECALL_NO_DELEGATION_GAS_DATA: list = []


class OutOfGasAt(Enum):
    """
    Enumeration of specific gas boundaries where OOG can occur.
    """

    EIP_2200_STIPEND = "oog_at_eip2200_stipend"
    EIP_2200_STIPEND_PLUS_1 = "oog_at_eip2200_stipend_plus_1"
    EXACT_GAS_MINUS_1 = "oog_at_exact_gas_minus_1"


class OutOfGasBoundary(Enum):
    """
    OOG boundary scenarios for call-type opcodes with 7702 delegation.

    For 7702 targets, there's ALWAYS a gap between static gas check and
    second check (delegation_cost + message_gas). All 4 scenarios test
    distinct boundaries.

    Gas check order:
    1. oog_before_target_access: access + transfer (if applicable) + memory.
       OOG with not enough for this check - no state access.
    2. oog_after_target_access: only enough for static check, state access
       reads target into BAL, not enough for anything else.
    3. oog_success_minus_1: exact gas minus 1. OOG here means target is in
       BAL, but we have enough information to calculate delegation cost
       AND the message call gas and not read if we don't have enough for
       both - delegation target NOT in BAL.
    4. success: target and delegation target both in BAL.

    OOG_SUCCESS_MINUS_1 tests that even when we have enough for delegation
    access cost, if we don't have enough for the total (missing subcall_gas),
    we don't read the delegation.
    """

    OOG_BEFORE_TARGET_ACCESS = "oog_before_target_access"
    OOG_AFTER_TARGET_ACCESS = "oog_after_target_access"
    OOG_SUCCESS_MINUS_1 = "oog_success_minus_1"
    SUCCESS = "success"


def print_gas_breakdown(
    test_name: str,
    oog_boundary: OutOfGasBoundary,
    target_is_warm: bool,
    delegation_is_warm: bool,
    value: int,
    memory_expansion: bool,
    ret_size: int,
    message_gas: int,
    intrinsic_cost: int,
    bytecode_cost: int,
    access_cost: int,
    transfer_cost: int,
    memory_cost: int,
    delegation_cost: int,
    static_gas_cost: int,
    second_check_cost: int,
    gas_limit: int,
    target_in_bal: bool,
    delegation_in_bal: bool,
    value_transferred: bool,
    result: dict,
    enabled: bool = True,
) -> None:
    """Collect gas breakdown data for aggregate reporting."""
    if not enabled:
        return

    # Calculate milestones
    milestone1 = (
        intrinsic_cost + bytecode_cost + static_gas_cost
    )  # Gas to access target
    milestone2 = (
        intrinsic_cost + bytecode_cost + second_check_cost
    )  # Gas to access delegation
    gap1 = gas_limit - milestone1  # Gap to milestone 1
    gap2 = gas_limit - milestone2  # Gap to milestone 2

    # Get actual gas from execution result
    gas_used = None
    if result and "gas_used" in result and result["gas_used"]:
        # Get gas from first transaction (block 0, tx 0)
        gas_used = result["gas_used"][0]["gas_used"]

    # Surplus = GasLimit - GasUsed (actual execution)
    # Will be 0 (OOG) or positive (leftover gas)
    surplus = (gas_limit - gas_used) if gas_used is not None else None

    # Store data for aggregate reporting
    _GAS_TRACKING_DATA.append(
        {
            "test_name": test_name,
            "boundary": oog_boundary.value,
            "target": "W" if target_is_warm else "C",
            "delegation": "W" if delegation_is_warm else "C",
            "value": value,
            "mem_expansion": ret_size,
            "intrinsic": intrinsic_cost,
            "bytecode": bytecode_cost,
            "access": access_cost,
            "transfer": transfer_cost,
            "mem_cost": memory_cost,
            "deleg_cost": delegation_cost,
            "msg_gas": message_gas,
            "static": static_gas_cost,
            "second": second_check_cost,
            "milestone1": milestone1,
            "milestone2": milestone2,
            "gap1": gap1,
            "gap2": gap2,
            "gas_limit": gas_limit,
            "gas_used": gas_used,
            "surplus": surplus,
            "tgt_bal": target_in_bal,
            "del_bal": delegation_in_bal,
            "xfer": value_transferred,
        }
    )


def print_gas_table(
    data: list,
    columns: list[dict],
    title: str,
    legend: str | None = None,
    enabled: bool = True,
) -> None:
    """
    Generic gas table printer with configurable columns.

    Args:
        data: List of dicts containing row data
        columns: List of column definitions, each with:
            - header: Column header text
            - key: Data key (str) or callable(row) -> value
            - justify: "left", "right", or "center" (default: "left")
            - width: Optional fixed width (default: auto)
            - no_wrap: Boolean (default: True)
            - style: Optional style string (default: None)
            - color_fn: Optional callable(value, row) -> colored_str
        title: Table title
        legend: Optional legend text to print after table
        enabled: Whether to print (default: True)
    """
    if not enabled or not data:
        return

    console = Console()
    console.print(f"\n[bold cyan]{title}[/bold cyan]\n")

    table = Table(show_lines=True, expand=True)

    # Add row number column
    table.add_column("#", justify="right", style="cyan", width=3, no_wrap=True)

    # Add configured columns
    for col in columns:
        table.add_column(
            col["header"],
            justify=col.get("justify", "left"),
            width=col.get("width"),
            no_wrap=col.get("no_wrap", True),
            style=col.get("style"),
            overflow=col.get("overflow", "fold"),
        )

    # Add rows
    for idx, row in enumerate(data, start=1):
        row_values = [str(idx)]
        for col in columns:
            # Get value from row using key or callable
            key = col["key"]
            if callable(key):
                value = key(row)
            else:
                value = row.get(key, "")

            # Apply color function if provided
            color_fn = col.get("color_fn")
            if color_fn:
                value_str = color_fn(value, row)
            else:
                value_str = str(value)

            row_values.append(value_str)

        table.add_row(*row_values)

    console.print(table)

    # Print legend if provided
    if legend:
        console.print(legend)


def print_aggregate_gas_table() -> None:
    """Print aggregate gas table for all collected test cases using rich."""
    if not _GAS_TRACKING_DATA:
        return

    # Color coding function for gap columns (can be negative, zero, or positive)
    def color_gap(value, row):
        if value < 0:
            return f"[red]{value}[/red]"
        elif value == 0:
            return f"[yellow]{value}[/yellow]"
        else:
            return f"[green]{value}[/green]"

    # Color coding function for surplus (0 or positive only)
    def color_surplus(value, row):
        if value is None:
            return "N/A"
        elif value == 0:
            return f"[yellow]{value}[/yellow]"
        else:
            return f"[green]{value}[/green]"

    # Column definitions
    columns = [
        {"header": "Test Parameters", "key": "test_name", "style": "dim", "width": 40, "no_wrap": False},
        {"header": "T", "key": "target", "justify": "center"},
        {"header": "D", "key": "delegation", "justify": "center"},
        {"header": "V", "key": "value", "justify": "center"},
        {"header": "MemExp", "key": "mem_expansion", "justify": "right"},
        {"header": "Intr (A)", "key": "intrinsic", "justify": "right"},
        {"header": "Byte (B)", "key": "bytecode", "justify": "right"},
        {"header": "TA (C)", "key": "access", "justify": "right"},
        {"header": "xfer (D)", "key": "transfer", "justify": "right"},
        {"header": "Mem (E)", "key": "mem_cost", "justify": "right"},
        {"header": "Static (F)", "key": "static", "justify": "right"},
        {"header": "M1", "key": "milestone1", "justify": "right"},
        {"header": "DA (G)", "key": "deleg_cost", "justify": "right"},
        {"header": "Second (H)", "key": "second", "justify": "right"},
        {"header": "M2", "key": "milestone2", "justify": "right"},
        {"header": "Msg (I)", "key": "msg_gas", "justify": "right"},
        {"header": "G1", "key": "gap1", "justify": "right", "color_fn": color_gap},
        {"header": "G2", "key": "gap2", "justify": "right", "color_fn": color_gap},
        {"header": "GasLimit", "key": "gas_limit", "justify": "right"},
        {"header": "GasUsed", "key": "gas_used", "justify": "right"},
        {"header": "Surplus", "key": "surplus", "justify": "right", "color_fn": color_surplus},
        {"header": "TgtBAL", "key": "tgt_bal", "justify": "center"},
        {"header": "DelBAL", "key": "del_bal", "justify": "center"},
        {"header": "ValXfer", "key": "xfer", "justify": "center"},
    ]

    # Legend
    legend = """
[bold cyan]═══ MILESTONE CONCEPT ═══[/bold cyan]
EIP-7702 CALL execution has two key milestones:
  [bold]M1 (Milestone 1)[/bold]: Gas needed to access target account
     • If GasLim ≥ M1 → Target appears in BAL
     • OOG_BEFORE_TARGET_ACCESS: G1 < 0
     • OOG_AFTER_TARGET_ACCESS: G1 ≥ 0, G2 < 0
  [bold]M2 (Milestone 2)[/bold]: Gas needed to access delegation
     • If GasLim ≥ M2 → Delegation appears in BAL
     • OOG_SUCCESS_MINUS_1: G2 = -1
     • SUCCESS: G2 ≥ 0

[bold]Legend:[/bold]
  [bold]Column Codes:[/bold]
    T/D/V/M: Target/Delegation warmness (W/C), Value (0/1), Memory (0/32)
    A: Intr - Intrinsic gas cost (base tx + access list)
    B: Byte - Bytecode cost (7 × PUSH for CALL params)

  [bold]Static Cost → Milestone 1:[/bold]
    C: TA - Target Access cost (cold=2600, warm=100)
    D: xfer - Transfer cost (0 or 9000 if value > 0)
    E: Mem - Memory expansion cost
    → F: Static - Static gas cost [C + D + E]
    → [cyan]M1: Milestone 1 [A + B + F] - Gas to access target[/cyan]

  [bold]Second Check → Milestone 2:[/bold]
    G: DA - Delegation Access cost (cold=2600, warm=100)
    → H: Second - Second check cost [G + F]
    → [cyan]M2: Milestone 2 [A + B + H] - Gas to access delegation[/cyan]
    Note: M2 = M1 + G (since H = G + F)

  [bold]Other Components:[/bold]
    I: Msg - Message gas passed to CALL (checked separately by EVM)
    GasLim: Transaction gas limit

  [bold]Gap Analysis:[/bold]
    [red]G1[/red]/[yellow]G1[/yellow]/[green]G1[/green] = GasLim - M1
      • [red]G1 < 0[/red]: OOG before target access
      • [yellow]G1 = 0[/yellow]: Exact boundary
      • [green]G1 > 0[/green]: Target accessed successfully
    [red]G2[/red]/[yellow]G2[/yellow]/[green]G2[/green] = GasLim - M2
      • [red]G2 < 0[/red]: OOG before delegation access
      • [yellow]G2 = 0[/yellow]: Exact boundary
      • [green]G2 > 0[/green]: Delegation accessed successfully

  [bold]Results:[/bold]
    TgtBAL: Target account in BAL (True/False)
    DelBAL: Delegation target in BAL (True/False)
    ValXfer: Value actually transferred (True/False)
"""

    print_gas_table(
        data=_GAS_TRACKING_DATA,
        columns=columns,
        title="AGGREGATE GAS BREAKDOWN - test_bal_call_7702_delegation_and_oog",
        legend=legend,
    )


def print_delegatecall_no_delegation_gas_breakdown(
    test_name: str,
    oog_boundary: OutOfGasBoundary,
    target_is_warm: bool,
    memory_expansion: bool,
    ret_size: int,
    intrinsic_cost: int,
    bytecode_cost: int,
    access_cost: int,
    memory_cost: int,
    static_gas_cost: int,
    gas_limit: int,
    target_in_bal: bool,
    result: dict,
    enabled: bool = True,
) -> None:
    """Collect gas breakdown data for DELEGATECALL (no delegation) test."""
    if not enabled:
        return

    # Calculate milestone (gas needed to reach target)
    milestone = intrinsic_cost + bytecode_cost + static_gas_cost

    # Get actual gas from execution result
    gas_used = None
    if result and "gas_used" in result and result["gas_used"]:
        # Get gas from first transaction (block 0, tx 0)
        gas_used = result["gas_used"][0]["gas_used"]

    # Gap to milestone = GasLimit - Milestone
    # Negative = OOG (not enough to reach milestone)
    # Zero = exact boundary
    # Positive = enough to reach milestone
    gap = gas_limit - milestone

    # Surplus = GasLimit - GasUsed (actual execution)
    # Will be 0 (OOG) or positive (leftover gas)
    surplus = (gas_limit - gas_used) if gas_used is not None else None

    # Store data for aggregate reporting
    _TEST_BAL_DELEGATECALL_NO_DELEGATION_GAS_DATA.append(
        {
            "test_name": test_name,
            "boundary": oog_boundary.value,
            "target": "W" if target_is_warm else "C",
            "mem_expansion": ret_size,
            "intrinsic": intrinsic_cost,
            "bytecode": bytecode_cost,
            "access": access_cost,
            "mem_cost": memory_cost,
            "static": static_gas_cost,
            "milestone": milestone,
            "gap": gap,
            "gas_limit": gas_limit,
            "gas_used": gas_used,
            "surplus": surplus,
            "tgt_bal": target_in_bal,
        }
    )


def print_delegatecall_no_delegation_aggregate_table() -> None:
    """Print aggregate gas table for DELEGATECALL (no delegation) test."""
    if not _TEST_BAL_DELEGATECALL_NO_DELEGATION_GAS_DATA:
        return

    # Color coding function for gap (can be negative, zero, or positive)
    def color_gap(value, row):
        if value < 0:
            return f"[red]{value}[/red]"
        elif value == 0:
            return f"[yellow]{value}[/yellow]"
        else:
            return f"[green]{value}[/green]"

    # Color coding function for surplus (0 or positive only)
    def color_surplus(value, row):
        if value is None:
            return "N/A"
        elif value == 0:
            return f"[yellow]{value}[/yellow]"
        else:
            return f"[green]{value}[/green]"

    # Column definitions
    columns = [
        {"header": "Test Parameters", "key": "test_name", "style": "dim", "width": 40, "no_wrap": False},
        {"header": "T", "key": "target", "justify": "center"},
        {"header": "MemExp", "key": "mem_expansion", "justify": "right"},
        {"header": "Intr (A)", "key": "intrinsic", "justify": "right"},
        {"header": "Byte (B)", "key": "bytecode", "justify": "right"},
        {"header": "Access (C)", "key": "access", "justify": "right"},
        {"header": "Mem (D)", "key": "mem_cost", "justify": "right"},
        {"header": "Static (E)", "key": "static", "justify": "right"},
        {"header": "M", "key": "milestone", "justify": "right"},
        {"header": "G", "key": "gap", "justify": "right", "color_fn": color_gap},
        {"header": "GasLimit", "key": "gas_limit", "justify": "right"},
        {"header": "GasUsed", "key": "gas_used", "justify": "right"},
        {"header": "Surplus", "key": "surplus", "justify": "right", "color_fn": color_surplus},
        {"header": "TgtBAL", "key": "tgt_bal", "justify": "center"},
    ]

    # Legend
    legend = """
[bold cyan]═══ MILESTONE CONCEPT ═══[/bold cyan]
DELEGATECALL (no delegation) has one milestone:
  [bold]M (Milestone)[/bold]: Gas needed to access target account
     • If GasLimit ≥ M → Target appears in BAL
     • OOG_BEFORE_TARGET_ACCESS: G < 0
     • SUCCESS: G ≥ 0

[bold]Legend:[/bold]
  [bold]Test Parameters:[/bold]
    T: Target warmness (W=warm, C=cold)
    MemExp: Memory expansion size (0 or 32 bytes)

  [bold]Gas Components:[/bold]
    A: Intr - Intrinsic gas cost (base tx + access list)
    B: Byte - Bytecode cost (6 × PUSH for DELEGATECALL params)
    C: Access - Target access cost (cold=2600, warm=100)
    D: Mem - Memory expansion cost
    → E: Static - Static gas cost [C + D]

  [bold]Milestone & Gap:[/bold]
    → [cyan]M (Milestone) = A + B + E[/cyan] - Gas needed to access target
    → [bold]G (Gap) = GasLimit - M[/bold]
      • [red]G < 0[/red]: OOG - not enough gas to reach milestone
      • [yellow]G = 0[/yellow]: Exact boundary
      • [green]G > 0[/green]: Enough gas to reach milestone

  [bold]Execution Results:[/bold]
    GasLimit: Transaction gas limit (what we send)
    GasUsed: Actual gas consumed by EVM execution
    [bold]Surplus = GasLimit - GasUsed[/bold]
      • [yellow]Surplus = 0[/yellow]: Out of gas (used all available)
      • [green]Surplus > 0[/green]: Leftover gas (successful execution)

  [bold]BAL Results:[/bold]
    TgtBAL: Target account in BAL (True/False)
"""

    print_gas_table(
        data=_TEST_BAL_DELEGATECALL_NO_DELEGATION_GAS_DATA,
        columns=columns,
        title="AGGREGATE GAS BREAKDOWN - test_bal_delegatecall_no_delegation_and_oog_before_target_access",
        legend=legend,
    )


@pytest.mark.parametrize(
    "out_of_gas_at",
    [
        OutOfGasAt.EIP_2200_STIPEND,
        OutOfGasAt.EIP_2200_STIPEND_PLUS_1,
        OutOfGasAt.EXACT_GAS_MINUS_1,
        None,  # no oog, successful sstore
    ],
    ids=lambda x: x.value if x else "successful_sstore",
)
def test_bal_sstore_and_oog(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
    fork: Fork,
    out_of_gas_at: OutOfGasAt | None,
) -> None:
    """
    Test BAL recording with SSTORE at various OOG boundaries and success.

    1. OOG at EIP-2200 stipend check & implicit SLOAD -> no BAL changes
    2. OOG post EIP-2200 stipend check & implicit SLOAD -> storage read in BAL
    3. OOG at exact gas minus 1 -> storage read in BAL
    4. exact gas (success) -> storage write in BAL
    """
    alice = pre.fund_eoa()
    gas_costs = fork.gas_costs()

    # Create contract that attempts SSTORE to cold storage slot 0x01
    storage_contract_code = Bytecode(Op.SSTORE(0x01, 0x42))

    storage_contract = pre.deploy_contract(code=storage_contract_code)

    intrinsic_gas_calculator = fork.transaction_intrinsic_cost_calculator()
    intrinsic_gas_cost = intrinsic_gas_calculator()

    # Costs:
    # - PUSH1 (value and slot) = G_VERY_LOW * 2
    # - SSTORE cold (to zero slot) = G_STORAGE_SET + G_COLD_SLOAD
    sload_cost = gas_costs.G_COLD_SLOAD
    sstore_cost = gas_costs.G_STORAGE_SET
    sstore_cold_cost = sstore_cost + sload_cost
    push_cost = gas_costs.G_VERY_LOW * 2
    stipend = gas_costs.G_CALL_STIPEND

    if out_of_gas_at == OutOfGasAt.EIP_2200_STIPEND:
        # 2300 after PUSHes (fails stipend check: 2300 <= 2300)
        tx_gas_limit = intrinsic_gas_cost + push_cost + stipend
    elif out_of_gas_at == OutOfGasAt.EIP_2200_STIPEND_PLUS_1:
        # 2301 after PUSHes (passes stipend, does SLOAD, fails charge_gas)
        tx_gas_limit = intrinsic_gas_cost + push_cost + stipend + 1
    elif out_of_gas_at == OutOfGasAt.EXACT_GAS_MINUS_1:
        # fail at charge_gas() at exact gas - 1 (boundary condition)
        tx_gas_limit = intrinsic_gas_cost + push_cost + sstore_cold_cost - 1
    else:
        # exact gas for successful SSTORE
        tx_gas_limit = intrinsic_gas_cost + push_cost + sstore_cold_cost

    tx = Transaction(
        sender=alice,
        to=storage_contract,
        gas_limit=tx_gas_limit,
    )

    # Storage read recorded only if we pass the stipend check and reach
    # implicit SLOAD (STIPEND_PLUS_1 and EXACT_GAS_MINUS_1)
    expect_storage_read = out_of_gas_at in (
        OutOfGasAt.EIP_2200_STIPEND_PLUS_1,
        OutOfGasAt.EXACT_GAS_MINUS_1,
    )
    expect_storage_write = out_of_gas_at is None

    block = Block(
        txs=[tx],
        expected_block_access_list=BlockAccessListExpectation(
            account_expectations={
                storage_contract: BalAccountExpectation(
                    storage_changes=[
                        BalStorageSlot(
                            slot=0x01,
                            slot_changes=[
                                BalStorageChange(tx_index=1, post_value=0x42)
                            ],
                        ),
                    ]
                    if expect_storage_write
                    else [],
                    storage_reads=[0x01] if expect_storage_read else [],
                )
            }
        ),
    )

    blockchain_test(
        pre=pre,
        blocks=[block],
        post={
            alice: Account(nonce=1),
            storage_contract: Account(
                storage={0x01: 0x42} if expect_storage_write else {}
            ),
        },
    )


@pytest.mark.parametrize(
    "fails_at_sload",
    [True, False],
    ids=["oog_at_sload", "successful_sload"],
)
def test_bal_sload_and_oog(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
    fork: Fork,
    fails_at_sload: bool,
) -> None:
    """
    Ensure BAL handles SLOAD and OOG during SLOAD appropriately.
    """
    alice = pre.fund_eoa()
    gas_costs = fork.gas_costs()

    # Create contract that attempts SLOAD from cold storage slot 0x01
    storage_contract_code = Bytecode(
        Op.PUSH1(0x01)  # Storage slot (cold)
        + Op.SLOAD  # Load value from slot - this will OOG
        + Op.STOP
    )

    storage_contract = pre.deploy_contract(code=storage_contract_code)

    intrinsic_gas_calculator = fork.transaction_intrinsic_cost_calculator()
    intrinsic_gas_cost = intrinsic_gas_calculator()

    # Costs:
    # - PUSH1 (slot) = G_VERY_LOW
    # - SLOAD cold = G_COLD_SLOAD
    push_cost = gas_costs.G_VERY_LOW
    sload_cold_cost = gas_costs.G_COLD_SLOAD
    tx_gas_limit = intrinsic_gas_cost + push_cost + sload_cold_cost

    if fails_at_sload:
        # subtract 1 gas to ensure OOG at SLOAD
        tx_gas_limit -= 1

    tx = Transaction(
        sender=alice,
        to=storage_contract,
        gas_limit=tx_gas_limit,
    )

    block = Block(
        txs=[tx],
        expected_block_access_list=BlockAccessListExpectation(
            account_expectations={
                storage_contract: BalAccountExpectation(
                    storage_reads=[] if fails_at_sload else [0x01],
                )
            }
        ),
    )

    blockchain_test(
        pre=pre,
        blocks=[block],
        post={
            alice: Account(nonce=1),
            storage_contract: Account(storage={}),
        },
    )


@pytest.mark.parametrize(
    "fails_at_balance",
    [True, False],
    ids=["oog_at_balance", "successful_balance"],
)
def test_bal_balance_and_oog(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
    fork: Fork,
    fails_at_balance: bool,
) -> None:
    """Ensure BAL handles BALANCE and OOG during BALANCE appropriately."""
    alice = pre.fund_eoa()
    bob = pre.fund_eoa()
    gas_costs = fork.gas_costs()

    # Create contract that attempts to check Bob's balance
    balance_checker_code = Bytecode(
        Op.PUSH20(bob)  # Bob's address
        + Op.BALANCE  # Check balance (cold access)
        + Op.STOP
    )

    balance_checker = pre.deploy_contract(code=balance_checker_code)

    intrinsic_gas_calculator = fork.transaction_intrinsic_cost_calculator()
    intrinsic_gas_cost = intrinsic_gas_calculator()

    # Costs:
    # - PUSH20 = G_VERY_LOW
    # - BALANCE cold = G_COLD_ACCOUNT_ACCESS
    push_cost = gas_costs.G_VERY_LOW
    balance_cold_cost = gas_costs.G_COLD_ACCOUNT_ACCESS
    tx_gas_limit = intrinsic_gas_cost + push_cost + balance_cold_cost

    if fails_at_balance:
        # subtract 1 gas to ensure OOG at BALANCE
        tx_gas_limit -= 1

    tx = Transaction(
        sender=alice,
        to=balance_checker,
        gas_limit=tx_gas_limit,
    )

    block = Block(
        txs=[tx],
        expected_block_access_list=BlockAccessListExpectation(
            account_expectations={
                balance_checker: BalAccountExpectation.empty(),
                # Bob should only appear in BAL if BALANCE succeeded
                **(
                    {bob: None}
                    if fails_at_balance
                    else {bob: BalAccountExpectation.empty()}
                ),
            }
        ),
    )

    blockchain_test(
        pre=pre,
        blocks=[block],
        post={
            alice: Account(nonce=1),
            bob: Account(),
            balance_checker: Account(),
        },
    )


@pytest.mark.parametrize(
    "fails_at_extcodesize",
    [True, False],
    ids=["oog_at_extcodesize", "successful_extcodesize"],
)
def test_bal_extcodesize_and_oog(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
    fork: Fork,
    fails_at_extcodesize: bool,
) -> None:
    """
    Ensure BAL handles EXTCODESIZE and OOG during EXTCODESIZE appropriately.
    """
    alice = pre.fund_eoa()
    gas_costs = fork.gas_costs()

    # Create target contract with some code
    target_contract = pre.deploy_contract(code=Bytecode(Op.STOP))

    # Create contract that checks target's code size
    codesize_checker_code = Bytecode(
        Op.PUSH20(target_contract)  # Target contract address
        + Op.EXTCODESIZE  # Check code size (cold access)
        + Op.STOP
    )

    codesize_checker = pre.deploy_contract(code=codesize_checker_code)

    intrinsic_gas_calculator = fork.transaction_intrinsic_cost_calculator()
    intrinsic_gas_cost = intrinsic_gas_calculator()

    # Costs:
    # - PUSH20 = G_VERY_LOW
    # - EXTCODESIZE cold = G_COLD_ACCOUNT_ACCESS
    push_cost = gas_costs.G_VERY_LOW
    extcodesize_cold_cost = gas_costs.G_COLD_ACCOUNT_ACCESS
    tx_gas_limit = intrinsic_gas_cost + push_cost + extcodesize_cold_cost

    if fails_at_extcodesize:
        # subtract 1 gas to ensure OOG at EXTCODESIZE
        tx_gas_limit -= 1

    tx = Transaction(
        sender=alice,
        to=codesize_checker,
        gas_limit=tx_gas_limit,
    )

    block = Block(
        txs=[tx],
        expected_block_access_list=BlockAccessListExpectation(
            account_expectations={
                codesize_checker: BalAccountExpectation.empty(),
                # Target should only appear if EXTCODESIZE succeeded
                **(
                    {target_contract: None}
                    if fails_at_extcodesize
                    else {target_contract: BalAccountExpectation.empty()}
                ),
            }
        ),
    )

    blockchain_test(
        pre=pre,
        blocks=[block],
        post={
            alice: Account(nonce=1),
            codesize_checker: Account(),
            target_contract: Account(),
        },
    )


@pytest.mark.parametrize(
    "oog_boundary",
    [OutOfGasBoundary.SUCCESS, OutOfGasBoundary.OOG_BEFORE_TARGET_ACCESS],
    ids=lambda x: x.value,
)
@pytest.mark.parametrize(
    "target_is_warm", [False, True], ids=["cold_target", "warm_target"]
)
@pytest.mark.parametrize(
    "target_is_empty", [False, True], ids=["existing_target", "empty_target"]
)
@pytest.mark.parametrize("value", [0, 1], ids=["no_value", "with_value"])
@pytest.mark.parametrize(
    "memory_expansion", [False, True], ids=["no_memory", "with_memory"]
)
def test_bal_call_no_delegation_and_oog_before_target_access(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
    fork: Fork,
    oog_boundary: OutOfGasBoundary,
    target_is_warm: bool,
    target_is_empty: bool,
    value: int,
    memory_expansion: bool,
) -> None:
    """
    CALL without 7702 delegation - test SUCCESS and OOG before target access.

    When target_is_warm=True, we use EIP-2930 tx access list to warm the
    target. Access list warming does NOT add to BAL - only EVM access does.
    """
    gas_costs = fork.gas_costs()
    alice = pre.fund_eoa()

    target = (
        pre.empty_account()
        if target_is_empty
        else pre.deploy_contract(code=Op.STOP)
    )

    ret_size = 32 if memory_expansion else 0

    call_code = Op.CALL(
        gas=0, address=target, value=value, ret_size=ret_size, ret_offset=0
    )
    caller = pre.deploy_contract(code=call_code, balance=value)

    access_list = (
        [AccessList(address=target, storage_keys=[])]
        if target_is_warm
        else None
    )

    intrinsic_cost = fork.transaction_intrinsic_cost_calculator()(
        access_list=access_list
    )

    bytecode_cost = gas_costs.G_VERY_LOW * 7

    access_cost = (
        gas_costs.G_WARM_ACCOUNT_ACCESS
        if target_is_warm
        else gas_costs.G_COLD_ACCOUNT_ACCESS
    )
    transfer_cost = gas_costs.G_CALL_VALUE if value > 0 else 0
    memory_cost = fork.memory_expansion_gas_calculator()(new_bytes=ret_size)

    # Create cost: only if value > 0 AND target is empty
    create_cost = (
        gas_costs.G_NEW_ACCOUNT if (value > 0 and target_is_empty) else 0
    )

    # static gas (before state access): access + transfer + memory
    static_gas_cost = access_cost + transfer_cost + memory_cost
    # second check includes create_cost
    second_check_cost = static_gas_cost + create_cost

    if oog_boundary == OutOfGasBoundary.OOG_BEFORE_TARGET_ACCESS:
        gas_limit = intrinsic_cost + bytecode_cost + static_gas_cost - 1
    else:  # SUCCESS
        gas_limit = intrinsic_cost + bytecode_cost + second_check_cost

    tx = Transaction(
        sender=alice,
        to=caller,
        gas_limit=gas_limit,
        access_list=access_list,
    )

    # BAL expectations
    account_expectations: Dict[Address, BalAccountExpectation | None]
    if oog_boundary == OutOfGasBoundary.OOG_BEFORE_TARGET_ACCESS:
        # Target NOT in BAL - we OOG before state access
        account_expectations = {
            caller: BalAccountExpectation.empty(),
            target: None,
        }
    elif value > 0:
        account_expectations = {
            caller: BalAccountExpectation(
                balance_changes=[BalBalanceChange(tx_index=1, post_balance=0)]
            ),
            target: BalAccountExpectation(
                balance_changes=[
                    BalBalanceChange(tx_index=1, post_balance=value)
                ]
            ),
        }
    else:
        account_expectations = {
            caller: BalAccountExpectation.empty(),
            target: BalAccountExpectation.empty(),
        }

    value_transferred = value > 0 and oog_boundary == OutOfGasBoundary.SUCCESS

    post_state: Dict[Address, Account | None] = {alice: Account(nonce=1)}

    if value_transferred:
        post_state[target] = Account(balance=value)
        post_state[caller] = Account(balance=0)
    else:
        post_state[caller] = Account(balance=value)
        post_state[target] = (
            Account.NONEXISTENT
            if target_is_empty
            else Account(balance=0, code=Op.STOP)
        )

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[tx],
                expected_block_access_list=BlockAccessListExpectation(
                    account_expectations=account_expectations
                ),
            )
        ],
        post=post_state,
    )


@pytest.mark.parametrize(
    "target_is_warm", [False, True], ids=["cold_target", "warm_target"]
)
@pytest.mark.parametrize(
    "memory_expansion", [False, True], ids=["no_memory", "with_memory"]
)
def test_bal_call_no_delegation_oog_after_target_access(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
    fork: Fork,
    target_is_warm: bool,
    memory_expansion: bool,
) -> None:
    """
    CALL without 7702 delegation - OOG after state access.

    When target_is_warm=True, uses EIP-2930 tx access list to warm the target.
    Access list warming does NOT add targets to BAL - only EVM access does.

    This test is only meaningful when there's a gap between gas check before
    state access and after state access. This only happens if create cost
    (empty target) and value transfer cost are both non-zero.

    Note:
        - target is always empty - required for create cost
        - value=1 (greater than 0) - required for create cost

    The create_cost (G_NEW_ACCOUNT = 25000) is charged only for value transfers
    to empty accounts, creating the gap tested here.

    """
    gas_costs = fork.gas_costs()
    alice = pre.fund_eoa()

    # empty target required for create_cost gap
    target = pre.empty_account()
    # value > 0 required for create_cost
    value = 1

    # memory expansion / no expansion
    ret_size = 32 if memory_expansion else 0

    # caller contract - no warmup code, we use tx access list instead
    call_code = Op.CALL(
        gas=0, address=target, value=value, ret_size=ret_size, ret_offset=0
    )
    caller = pre.deploy_contract(code=call_code, balance=value)

    # Access list for warming target (if needed)
    access_list = (
        [AccessList(address=target, storage_keys=[])]
        if target_is_warm
        else None
    )

    intrinsic_cost = fork.transaction_intrinsic_cost_calculator()(
        access_list=access_list
    )

    # Bytecode cost: 7 pushes for Op.CALL (no warmup code)
    bytecode_cost = gas_costs.G_VERY_LOW * 7

    # Access cost for CALL - warm if in tx access list
    access_cost = (
        gas_costs.G_WARM_ACCOUNT_ACCESS
        if target_is_warm
        else gas_costs.G_COLD_ACCOUNT_ACCESS
    )
    transfer_cost = gas_costs.G_CALL_VALUE  # value > 0, so always charged
    memory_cost = fork.memory_expansion_gas_calculator()(new_bytes=ret_size)

    # static gas cost (before state access): access + transfer + memory
    static_gas_cost = access_cost + transfer_cost + memory_cost

    # Pass static check, fail at second check due to create cost
    # (create_cost = G_NEW_ACCOUNT = 25000 for empty target + value > 0)
    gas_limit = intrinsic_cost + bytecode_cost + static_gas_cost

    tx = Transaction(
        sender=alice,
        to=caller,
        gas_limit=gas_limit,
        access_list=access_list,
    )

    # Target is always in BAL after state access but value transfer fails
    # (no balance changes)
    account_expectations: Dict[Address, BalAccountExpectation | None] = {
        caller: BalAccountExpectation.empty(),
        target: BalAccountExpectation.empty(),
    }

    post_state = {
        alice: Account(nonce=1),
        caller: Account(balance=value),
        target: Account.NONEXISTENT,
    }

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[tx],
                expected_block_access_list=BlockAccessListExpectation(
                    account_expectations=account_expectations
                ),
            )
        ],
        post=post_state,
    )


@pytest.mark.parametrize(
    "oog_boundary",
    list(OutOfGasBoundary),
    ids=lambda x: x.value,
)
@pytest.mark.parametrize(
    "target_is_warm", [False, True], ids=["cold_target", "warm_target"]
)
@pytest.mark.parametrize(
    "delegation_is_warm",
    [False, True],
    ids=["cold_delegation", "warm_delegation"],
)
@pytest.mark.parametrize("value", [0, 1], ids=["no_value", "with_value"])
@pytest.mark.parametrize(
    "memory_expansion", [False, True], ids=["no_memory", "with_memory"]
)
def test_bal_call_7702_delegation_and_oog(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
    fork: Fork,
    oog_boundary: OutOfGasBoundary,
    target_is_warm: bool,
    delegation_is_warm: bool,
    value: int,
    memory_expansion: bool,
) -> None:
    """
    CALL with 7702 delegation - test all OOG boundaries.

    When target_is_warm or delegation_is_warm, we use EIP-2930 tx access list.
    Access list warming does NOT add targets to BAL - only EVM access does.
    """
    gas_costs = fork.gas_costs()
    alice = pre.fund_eoa()

    delegation_target = pre.deploy_contract(code=Op.STOP)
    target = pre.fund_eoa(amount=0, delegation=delegation_target)

    # memory expansion / no expansion
    ret_size = 32 if memory_expansion else 0

    # Use gas=1 to test that when we have enough for delegation_cost but not
    # enough for delegation_cost + message_gas, we still don't read the
    # delegation (the EVM does a static check upfront)
    message_gas = 1
    call_code = Op.CALL(
        gas=message_gas,
        address=target,
        value=value,
        ret_size=ret_size,
        ret_offset=0,
    )
    caller = pre.deploy_contract(code=call_code, balance=value)

    # Build access list for warming
    access_list: list[AccessList] = []
    if target_is_warm:
        access_list.append(AccessList(address=target, storage_keys=[]))
    if delegation_is_warm:
        access_list.append(
            AccessList(address=delegation_target, storage_keys=[])
        )
    access_list_or_none = access_list if access_list else None

    intrinsic_cost = fork.transaction_intrinsic_cost_calculator()(
        access_list=access_list_or_none
    )

    bytecode_cost = gas_costs.G_VERY_LOW * 7

    access_cost = (
        gas_costs.G_WARM_ACCOUNT_ACCESS
        if target_is_warm
        else gas_costs.G_COLD_ACCOUNT_ACCESS
    )
    transfer_cost = gas_costs.G_CALL_VALUE if value > 0 else 0
    memory_cost = fork.memory_expansion_gas_calculator()(new_bytes=ret_size)
    delegation_cost = (
        gas_costs.G_WARM_ACCOUNT_ACCESS
        if delegation_is_warm
        else gas_costs.G_COLD_ACCOUNT_ACCESS
    )

    static_gas_cost = access_cost + transfer_cost + memory_cost
    # The EVM's second check cost is static_gas + delegation_cost.
    # When gas_left == extra_gas, calculate_message_call_gas caps the
    # provided gas to 0, so message_gas doesn't contribute to the cost.
    second_check_cost = static_gas_cost + delegation_cost

    if oog_boundary == OutOfGasBoundary.OOG_BEFORE_TARGET_ACCESS:
        gas_limit = intrinsic_cost + bytecode_cost + static_gas_cost - 1
    elif oog_boundary == OutOfGasBoundary.OOG_AFTER_TARGET_ACCESS:
        # Enough for static_gas only - not enough for delegation_cost.
        gas_limit = intrinsic_cost + bytecode_cost + static_gas_cost
    elif oog_boundary == OutOfGasBoundary.OOG_SUCCESS_MINUS_1:
        # One less than second_check_cost. This hits the early return path
        # in calculate_message_call_gas where message_gas IS added to cost,
        # causing OOG. Proves the exact boundary of the second check.
        gas_limit = intrinsic_cost + bytecode_cost + second_check_cost - 1
    else:
        gas_limit = intrinsic_cost + bytecode_cost + second_check_cost

    tx = Transaction(
        sender=alice,
        to=caller,
        gas_limit=gas_limit,
        access_list=access_list_or_none,
    )

    # Access list warming does NOT add to BAL - only EVM execution does
    if oog_boundary == OutOfGasBoundary.OOG_BEFORE_TARGET_ACCESS:
        target_in_bal = False
        delegation_in_bal = False
    elif oog_boundary in (
        OutOfGasBoundary.OOG_AFTER_TARGET_ACCESS,
        OutOfGasBoundary.OOG_SUCCESS_MINUS_1,
    ):
        # Both cases: target accessed but not enough gas for full call
        # so delegation is NOT read (static check optimization)
        target_in_bal = True
        delegation_in_bal = False
    else:
        target_in_bal = True
        delegation_in_bal = True

    value_transferred = value > 0 and oog_boundary == OutOfGasBoundary.SUCCESS

    account_expectations: Dict[Address, BalAccountExpectation | None] = {
        caller: (
            BalAccountExpectation(
                balance_changes=[BalBalanceChange(tx_index=1, post_balance=0)]
            )
            if value_transferred
            else BalAccountExpectation.empty()
        ),
        delegation_target: (
            BalAccountExpectation.empty() if delegation_in_bal else None
        ),
    }

    if target_in_bal:
        if value_transferred:
            account_expectations[target] = BalAccountExpectation(
                balance_changes=[
                    BalBalanceChange(tx_index=1, post_balance=value)
                ]
            )
        else:
            account_expectations[target] = BalAccountExpectation.empty()
    else:
        account_expectations[target] = None

    # Post-state balance checks verify value transfer only happened on success
    post_state: Dict[Address, Account] = {alice: Account(nonce=1)}
    if value > 0:
        post_state[target] = Account(balance=value if value_transferred else 0)
        post_state[caller] = Account(balance=0 if value_transferred else value)

    test_instance = blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[tx],
                expected_block_access_list=BlockAccessListExpectation(
                    account_expectations=account_expectations
                ),
            )
        ],
        post=post_state,
    )

    # Track gas breakdown AFTER blockchain_test() executes
    target_label = "warm_target" if target_is_warm else "cold_target"
    delegation_label = (
        "warm_delegation" if delegation_is_warm else "cold_delegation"
    )
    value_label = "with_value" if value > 0 else "no_value"
    memory_label = "with_memory" if memory_expansion else "no_memory"
    param_combo = f"{oog_boundary.value}, {target_label}, {delegation_label}, {value_label}, {memory_label}"

    print_gas_breakdown(
        test_name=param_combo,
        oog_boundary=oog_boundary,
        target_is_warm=target_is_warm,
        delegation_is_warm=delegation_is_warm,
        value=value,
        memory_expansion=memory_expansion,
        ret_size=ret_size,
        message_gas=message_gas,
        intrinsic_cost=intrinsic_cost,
        bytecode_cost=bytecode_cost,
        access_cost=access_cost,
        transfer_cost=transfer_cost,
        memory_cost=memory_cost,
        delegation_cost=delegation_cost,
        static_gas_cost=static_gas_cost,
        second_check_cost=second_check_cost,
        gas_limit=gas_limit,
        target_in_bal=target_in_bal,
        delegation_in_bal=delegation_in_bal,
        value_transferred=value_transferred,
        result=test_instance.get_result(),
    )


@pytest.mark.parametrize(
    "oog_boundary",
    [OutOfGasBoundary.SUCCESS, OutOfGasBoundary.OOG_BEFORE_TARGET_ACCESS],
    ids=lambda x: x.value,
)
@pytest.mark.parametrize(
    "target_is_warm", [False, True], ids=["cold_target", "warm_target"]
)
@pytest.mark.parametrize(
    "memory_expansion", [False, True], ids=["no_memory", "with_memory"]
)
def test_bal_delegatecall_no_delegation_and_oog_before_target_access(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
    fork: Fork,
    oog_boundary: OutOfGasBoundary,
    target_is_warm: bool,
    memory_expansion: bool,
) -> None:
    """
    DELEGATECALL without 7702 delegation - test SUCCESS and OOG boundaries.

    When target_is_warm=True, we use EIP-2930 tx access list to warm the
    target. Access list warming does NOT add to BAL - only EVM access does.
    """
    alice = pre.fund_eoa()
    gas_costs = fork.gas_costs()

    target = pre.deploy_contract(code=Op.STOP)

    ret_size = 32 if memory_expansion else 0
    ret_offset = 0

    delegatecall_code = Op.DELEGATECALL(
        address=target,
        gas=0,
        ret_size=ret_size,
        ret_offset=ret_offset,
    )

    caller = pre.deploy_contract(code=delegatecall_code)

    access_list = (
        [AccessList(address=target, storage_keys=[])]
        if target_is_warm
        else None
    )

    intrinsic_cost = fork.transaction_intrinsic_cost_calculator()(
        access_list=access_list
    )

    # 6 pushes: retSize, retOffset, argsSize, argsOffset, address, gas
    bytecode_cost = gas_costs.G_VERY_LOW * 6

    access_cost = (
        gas_costs.G_WARM_ACCOUNT_ACCESS
        if target_is_warm
        else gas_costs.G_COLD_ACCOUNT_ACCESS
    )

    memory_cost = fork.memory_expansion_gas_calculator()(new_bytes=ret_size)

    # static gas (before state access) == second check (no delegation cost)
    static_gas_cost = access_cost + memory_cost

    if oog_boundary == OutOfGasBoundary.OOG_BEFORE_TARGET_ACCESS:
        gas_limit = intrinsic_cost + bytecode_cost + static_gas_cost - 1
        target_in_bal = False
    else:  # SUCCESS
        gas_limit = intrinsic_cost + bytecode_cost + static_gas_cost
        target_in_bal = True

    tx = Transaction(
        sender=alice,
        to=caller,
        gas_limit=gas_limit,
        access_list=access_list,
    )

    # BAL expectations
    account_expectations: Dict[Address, BalAccountExpectation | None]
    if oog_boundary == OutOfGasBoundary.OOG_BEFORE_TARGET_ACCESS:
        # Target NOT in BAL - we OOG before state access
        account_expectations = {
            caller: BalAccountExpectation.empty(),
            target: None,
        }
    else:  # SUCCESS - target in BAL
        account_expectations = {
            caller: BalAccountExpectation.empty(),
            target: BalAccountExpectation.empty(),
        }

    test_instance = blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[tx],
                expected_block_access_list=BlockAccessListExpectation(
                    account_expectations=account_expectations
                ),
            )
        ],
        post={alice: Account(nonce=1)},
    )

    # Track gas breakdown AFTER blockchain_test() executes
    target_label = "warm_target" if target_is_warm else "cold_target"
    memory_label = "with_memory" if memory_expansion else "no_memory"
    param_combo = f"{oog_boundary.value}, {target_label}, {memory_label}"

    print_delegatecall_no_delegation_gas_breakdown(
        test_name=param_combo,
        oog_boundary=oog_boundary,
        target_is_warm=target_is_warm,
        memory_expansion=memory_expansion,
        ret_size=ret_size,
        intrinsic_cost=intrinsic_cost,
        bytecode_cost=bytecode_cost,
        access_cost=access_cost,
        memory_cost=memory_cost,
        static_gas_cost=static_gas_cost,
        gas_limit=gas_limit,
        target_in_bal=target_in_bal,
        result=test_instance.get_result(),
    )


@pytest.mark.parametrize(
    "oog_boundary",
    list(OutOfGasBoundary),
    ids=lambda x: x.value,
)
@pytest.mark.parametrize(
    "target_is_warm", [False, True], ids=["cold_target", "warm_target"]
)
@pytest.mark.parametrize(
    "delegation_is_warm",
    [False, True],
    ids=["cold_delegation", "warm_delegation"],
)
@pytest.mark.parametrize(
    "memory_expansion", [False, True], ids=["no_memory", "with_memory"]
)
def test_bal_delegatecall_7702_delegation_and_oog(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
    fork: Fork,
    oog_boundary: OutOfGasBoundary,
    target_is_warm: bool,
    delegation_is_warm: bool,
    memory_expansion: bool,
) -> None:
    """
    DELEGATECALL with 7702 delegation - test all OOG boundaries.

    When target_is_warm or delegation_is_warm, we use EIP-2930 tx access list.
    Access list warming does NOT add targets to BAL - only EVM access does.

    For 7702 delegation, there's ALWAYS a gap between static gas and
    second check (delegation_cost) - all 3 scenarios produce distinct
    behaviors.
    """
    alice = pre.fund_eoa()
    gas_costs = fork.gas_costs()

    delegation_target = pre.deploy_contract(code=Op.STOP)
    target = pre.fund_eoa(amount=0, delegation=delegation_target)

    # memory expansion / no expansion
    ret_size = 32 if memory_expansion else 0
    ret_offset = 0

    # Use gas=1 to test that when we have enough for delegation_cost but not
    # enough for delegation_cost + message_gas, we still don't read the
    # delegation (the EVM does a static check upfront)
    message_gas = 1
    delegatecall_code = Op.DELEGATECALL(
        address=target,
        gas=message_gas,
        ret_size=ret_size,
        ret_offset=ret_offset,
    )

    caller = pre.deploy_contract(code=delegatecall_code)

    # Build access list for warming
    access_list: list[AccessList] = []
    if target_is_warm:
        access_list.append(AccessList(address=target, storage_keys=[]))
    if delegation_is_warm:
        access_list.append(
            AccessList(address=delegation_target, storage_keys=[])
        )
    access_list_or_none = access_list if access_list else None

    intrinsic_cost = fork.transaction_intrinsic_cost_calculator()(
        access_list=access_list_or_none
    )

    bytecode_cost = gas_costs.G_VERY_LOW * 6

    access_cost = (
        gas_costs.G_WARM_ACCOUNT_ACCESS
        if target_is_warm
        else gas_costs.G_COLD_ACCOUNT_ACCESS
    )
    memory_cost = fork.memory_expansion_gas_calculator()(new_bytes=ret_size)
    delegation_cost = (
        gas_costs.G_WARM_ACCOUNT_ACCESS
        if delegation_is_warm
        else gas_costs.G_COLD_ACCOUNT_ACCESS
    )

    static_gas_cost = access_cost + memory_cost
    # The EVM's second check cost is static_gas + delegation_cost.
    # When gas_left == extra_gas, calculate_message_call_gas caps the
    # provided gas to 0, so message_gas doesn't contribute to the cost.
    second_check_cost = static_gas_cost + delegation_cost

    if oog_boundary == OutOfGasBoundary.OOG_BEFORE_TARGET_ACCESS:
        gas_limit = intrinsic_cost + bytecode_cost + static_gas_cost - 1
    elif oog_boundary == OutOfGasBoundary.OOG_AFTER_TARGET_ACCESS:
        # Enough for static_gas only - not enough for delegation_cost.
        gas_limit = intrinsic_cost + bytecode_cost + static_gas_cost
    elif oog_boundary == OutOfGasBoundary.OOG_SUCCESS_MINUS_1:
        # One less than second_check_cost. This hits the early return path
        # in calculate_message_call_gas where message_gas IS added to cost,
        # causing OOG. Proves the exact boundary of the second check.
        gas_limit = intrinsic_cost + bytecode_cost + second_check_cost - 1
    else:
        gas_limit = intrinsic_cost + bytecode_cost + second_check_cost

    tx = Transaction(
        sender=alice,
        to=caller,
        gas_limit=gas_limit,
        access_list=access_list_or_none,
    )

    # Access list warming does NOT add to BAL - only EVM execution does
    if oog_boundary == OutOfGasBoundary.OOG_BEFORE_TARGET_ACCESS:
        target_in_bal = False
        delegation_in_bal = False
    elif oog_boundary in (
        OutOfGasBoundary.OOG_AFTER_TARGET_ACCESS,
        OutOfGasBoundary.OOG_SUCCESS_MINUS_1,
    ):
        # Both cases: target accessed but not enough gas for full call
        # so delegation is NOT read (static check optimization)
        target_in_bal = True
        delegation_in_bal = False
    else:
        target_in_bal = True
        delegation_in_bal = True

    account_expectations: Dict[Address, BalAccountExpectation | None] = {
        caller: BalAccountExpectation.empty(),
        delegation_target: (
            BalAccountExpectation.empty() if delegation_in_bal else None
        ),
    }

    if target_in_bal:
        account_expectations[target] = BalAccountExpectation.empty()
    else:
        account_expectations[target] = None

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[tx],
                expected_block_access_list=BlockAccessListExpectation(
                    account_expectations=account_expectations
                ),
            )
        ],
        post={alice: Account(nonce=1)},
    )


@pytest.mark.parametrize(
    "oog_boundary",
    [OutOfGasBoundary.SUCCESS, OutOfGasBoundary.OOG_BEFORE_TARGET_ACCESS],
    ids=lambda x: x.value,
)
@pytest.mark.parametrize(
    "target_is_warm", [False, True], ids=["cold_target", "warm_target"]
)
@pytest.mark.parametrize("value", [0, 1], ids=["no_value", "with_value"])
@pytest.mark.parametrize(
    "memory_expansion", [False, True], ids=["no_memory", "with_memory"]
)
def test_bal_callcode_no_delegation_and_oog_before_target_access(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
    fork: Fork,
    oog_boundary: OutOfGasBoundary,
    target_is_warm: bool,
    value: int,
    memory_expansion: bool,
) -> None:
    """
    CALLCODE without 7702 delegation - test SUCCESS and OOG boundaries.

    When target_is_warm=True, we use EIP-2930 tx access list to warm the
    target. Access list warming does NOT add to BAL - only EVM access does.
    CALLCODE has no balance transfer to target (runs in caller's context).
    """
    gas_costs = fork.gas_costs()
    alice = pre.fund_eoa()

    target = pre.deploy_contract(code=Op.STOP)

    ret_size = 32 if memory_expansion else 0

    callcode_code = Op.CALLCODE(
        gas=0, address=target, value=value, ret_size=ret_size, ret_offset=0
    )
    caller = pre.deploy_contract(code=callcode_code, balance=value)

    access_list = (
        [AccessList(address=target, storage_keys=[])]
        if target_is_warm
        else None
    )

    intrinsic_cost = fork.transaction_intrinsic_cost_calculator()(
        access_list=access_list
    )

    bytecode_cost = gas_costs.G_VERY_LOW * 7

    access_cost = (
        gas_costs.G_WARM_ACCOUNT_ACCESS
        if target_is_warm
        else gas_costs.G_COLD_ACCOUNT_ACCESS
    )
    transfer_cost = gas_costs.G_CALL_VALUE if value > 0 else 0
    memory_cost = fork.memory_expansion_gas_calculator()(new_bytes=ret_size)

    # static gas: access + transfer + memory (== second check, no delegation)
    static_gas_cost = access_cost + transfer_cost + memory_cost

    if oog_boundary == OutOfGasBoundary.OOG_BEFORE_TARGET_ACCESS:
        gas_limit = intrinsic_cost + bytecode_cost + static_gas_cost - 1
    else:  # SUCCESS
        gas_limit = intrinsic_cost + bytecode_cost + static_gas_cost

    tx = Transaction(
        sender=alice,
        to=caller,
        gas_limit=gas_limit,
        access_list=access_list,
    )

    # BAL expectations
    account_expectations: Dict[Address, BalAccountExpectation | None]
    if oog_boundary == OutOfGasBoundary.OOG_BEFORE_TARGET_ACCESS:
        # Target NOT in BAL - we OOG before state access
        account_expectations = {
            caller: BalAccountExpectation.empty(),
            target: None,
        }
    else:  # SUCCESS - target in BAL (no balance changes, CALLCODE no transfer)
        account_expectations = {
            caller: BalAccountExpectation.empty(),
            target: BalAccountExpectation.empty(),
        }

    # Post-state: CALLCODE runs in caller's context, so value transfer is
    # caller-to-caller (net-zero). Caller keeps its balance regardless.
    post_state: Dict[Address, Account] = {
        alice: Account(nonce=1),
        caller: Account(balance=value),
    }

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[tx],
                expected_block_access_list=BlockAccessListExpectation(
                    account_expectations=account_expectations
                ),
            )
        ],
        post=post_state,
    )


@pytest.mark.parametrize(
    "oog_boundary",
    list(OutOfGasBoundary),
    ids=lambda x: x.value,
)
@pytest.mark.parametrize(
    "target_is_warm", [False, True], ids=["cold_target", "warm_target"]
)
@pytest.mark.parametrize(
    "delegation_is_warm",
    [False, True],
    ids=["cold_delegation", "warm_delegation"],
)
@pytest.mark.parametrize("value", [0, 1], ids=["no_value", "with_value"])
@pytest.mark.parametrize(
    "memory_expansion", [False, True], ids=["no_memory", "with_memory"]
)
def test_bal_callcode_7702_delegation_and_oog(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
    fork: Fork,
    oog_boundary: OutOfGasBoundary,
    target_is_warm: bool,
    delegation_is_warm: bool,
    value: int,
    memory_expansion: bool,
) -> None:
    """
    CALLCODE with 7702 delegation - test all OOG boundaries.

    When target_is_warm or delegation_is_warm, we use EIP-2930 tx access list.
    Access list warming does NOT add targets to BAL - only EVM access does.

    For 7702 delegation, there's ALWAYS a gap between static gas and
    second check (delegation_cost) - all 3 scenarios produce distinct
    behaviors.
    """
    gas_costs = fork.gas_costs()
    alice = pre.fund_eoa()

    delegation_target = pre.deploy_contract(code=Op.STOP)
    target = pre.fund_eoa(amount=0, delegation=delegation_target)

    # memory expansion / no expansion
    ret_size = 32 if memory_expansion else 0

    # Use gas=1 to test that when we have enough for delegation_cost but not
    # enough for delegation_cost + message_gas, we still don't read the
    # delegation (the EVM does a static check upfront)
    message_gas = 1
    callcode_code = Op.CALLCODE(
        gas=message_gas,
        address=target,
        value=value,
        ret_size=ret_size,
        ret_offset=0,
    )
    caller = pre.deploy_contract(code=callcode_code, balance=value)

    # Build access list for warming
    access_list: list[AccessList] = []
    if target_is_warm:
        access_list.append(AccessList(address=target, storage_keys=[]))
    if delegation_is_warm:
        access_list.append(
            AccessList(address=delegation_target, storage_keys=[])
        )
    access_list_or_none = access_list if access_list else None

    intrinsic_cost = fork.transaction_intrinsic_cost_calculator()(
        access_list=access_list_or_none
    )

    bytecode_cost = gas_costs.G_VERY_LOW * 7

    access_cost = (
        gas_costs.G_WARM_ACCOUNT_ACCESS
        if target_is_warm
        else gas_costs.G_COLD_ACCOUNT_ACCESS
    )
    transfer_cost = gas_costs.G_CALL_VALUE if value > 0 else 0
    memory_cost = fork.memory_expansion_gas_calculator()(new_bytes=ret_size)
    delegation_cost = (
        gas_costs.G_WARM_ACCOUNT_ACCESS
        if delegation_is_warm
        else gas_costs.G_COLD_ACCOUNT_ACCESS
    )

    static_gas_cost = access_cost + transfer_cost + memory_cost
    # The EVM's second check cost is static_gas + delegation_cost.
    # When gas_left == extra_gas, calculate_message_call_gas caps the
    # provided gas to 0, so message_gas doesn't contribute to the cost.
    second_check_cost = static_gas_cost + delegation_cost

    if oog_boundary == OutOfGasBoundary.OOG_BEFORE_TARGET_ACCESS:
        gas_limit = intrinsic_cost + bytecode_cost + static_gas_cost - 1
    elif oog_boundary == OutOfGasBoundary.OOG_AFTER_TARGET_ACCESS:
        # Enough for static_gas only - not enough for delegation_cost.
        gas_limit = intrinsic_cost + bytecode_cost + static_gas_cost
    elif oog_boundary == OutOfGasBoundary.OOG_SUCCESS_MINUS_1:
        # One less than second_check_cost. This hits the early return path
        # in calculate_message_call_gas where message_gas IS added to cost,
        # causing OOG. Proves the exact boundary of the second check.
        gas_limit = intrinsic_cost + bytecode_cost + second_check_cost - 1
    else:
        gas_limit = intrinsic_cost + bytecode_cost + second_check_cost

    tx = Transaction(
        sender=alice,
        to=caller,
        gas_limit=gas_limit,
        access_list=access_list_or_none,
    )

    # Access list warming does NOT add to BAL - only EVM execution does
    if oog_boundary == OutOfGasBoundary.OOG_BEFORE_TARGET_ACCESS:
        target_in_bal = False
        delegation_in_bal = False
    elif oog_boundary in (
        OutOfGasBoundary.OOG_AFTER_TARGET_ACCESS,
        OutOfGasBoundary.OOG_SUCCESS_MINUS_1,
    ):
        # Both cases: target accessed but not enough gas for full call
        # so delegation is NOT read (static check optimization)
        target_in_bal = True
        delegation_in_bal = False
    else:
        target_in_bal = True
        delegation_in_bal = True

    account_expectations: Dict[Address, BalAccountExpectation | None] = {
        caller: BalAccountExpectation.empty(),
        delegation_target: (
            BalAccountExpectation.empty() if delegation_in_bal else None
        ),
    }

    if target_in_bal:
        account_expectations[target] = BalAccountExpectation.empty()
    else:
        account_expectations[target] = None

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[tx],
                expected_block_access_list=BlockAccessListExpectation(
                    account_expectations=account_expectations
                ),
            )
        ],
        post={alice: Account(nonce=1)},
    )


@pytest.mark.parametrize(
    "oog_boundary",
    [OutOfGasBoundary.SUCCESS, OutOfGasBoundary.OOG_BEFORE_TARGET_ACCESS],
    ids=lambda x: x.value,
)
@pytest.mark.parametrize(
    "target_is_warm", [False, True], ids=["cold_target", "warm_target"]
)
@pytest.mark.parametrize(
    "memory_expansion", [False, True], ids=["no_memory", "with_memory"]
)
def test_bal_staticcall_no_delegation_and_oog_before_target_access(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
    fork: Fork,
    oog_boundary: OutOfGasBoundary,
    target_is_warm: bool,
    memory_expansion: bool,
) -> None:
    """
    STATICCALL without 7702 delegation - test SUCCESS and OOG boundaries.

    When target_is_warm=True, we use EIP-2930 tx access list to warm the
    target. Access list warming does NOT add to BAL - only EVM access does.
    """
    alice = pre.fund_eoa()
    gas_costs = fork.gas_costs()

    target = pre.deploy_contract(code=Op.STOP)

    ret_size = 32 if memory_expansion else 0
    ret_offset = 0

    staticcall_code = Op.STATICCALL(
        address=target,
        gas=0,
        ret_size=ret_size,
        ret_offset=ret_offset,
    )

    caller = pre.deploy_contract(code=staticcall_code)

    access_list = (
        [AccessList(address=target, storage_keys=[])]
        if target_is_warm
        else None
    )

    intrinsic_cost = fork.transaction_intrinsic_cost_calculator()(
        access_list=access_list
    )

    # 6 pushes: retSize, retOffset, argsSize, argsOffset, address, gas
    bytecode_cost = gas_costs.G_VERY_LOW * 6

    access_cost = (
        gas_costs.G_WARM_ACCOUNT_ACCESS
        if target_is_warm
        else gas_costs.G_COLD_ACCOUNT_ACCESS
    )

    memory_cost = fork.memory_expansion_gas_calculator()(new_bytes=ret_size)

    # static gas (before state access) == second check (no delegation cost)
    static_gas_cost = access_cost + memory_cost

    if oog_boundary == OutOfGasBoundary.OOG_BEFORE_TARGET_ACCESS:
        gas_limit = intrinsic_cost + bytecode_cost + static_gas_cost - 1
    else:  # SUCCESS
        gas_limit = intrinsic_cost + bytecode_cost + static_gas_cost

    tx = Transaction(
        sender=alice,
        to=caller,
        gas_limit=gas_limit,
        access_list=access_list,
    )

    # BAL expectations
    account_expectations: Dict[Address, BalAccountExpectation | None]
    if oog_boundary == OutOfGasBoundary.OOG_BEFORE_TARGET_ACCESS:
        # Target NOT in BAL - we OOG before state access
        account_expectations = {
            caller: BalAccountExpectation.empty(),
            target: None,
        }
    else:  # SUCCESS - target in BAL
        account_expectations = {
            caller: BalAccountExpectation.empty(),
            target: BalAccountExpectation.empty(),
        }

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[tx],
                expected_block_access_list=BlockAccessListExpectation(
                    account_expectations=account_expectations
                ),
            )
        ],
        post={alice: Account(nonce=1)},
    )


@pytest.mark.parametrize(
    "oog_boundary",
    list(OutOfGasBoundary),
    ids=lambda x: x.value,
)
@pytest.mark.parametrize(
    "target_is_warm", [False, True], ids=["cold_target", "warm_target"]
)
@pytest.mark.parametrize(
    "delegation_is_warm",
    [False, True],
    ids=["cold_delegation", "warm_delegation"],
)
@pytest.mark.parametrize(
    "memory_expansion", [False, True], ids=["no_memory", "with_memory"]
)
def test_bal_staticcall_7702_delegation_and_oog(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
    fork: Fork,
    oog_boundary: OutOfGasBoundary,
    target_is_warm: bool,
    delegation_is_warm: bool,
    memory_expansion: bool,
) -> None:
    """
    STATICCALL with 7702 delegation - test all OOG boundaries.

    When target_is_warm or delegation_is_warm, we use EIP-2930 tx access list.
    Access list warming does NOT add targets to BAL - only EVM access does.

    For 7702 delegation, there's ALWAYS a gap between static gas and
    second check (delegation_cost) - all 3 scenarios produce distinct
    behaviors.
    """
    alice = pre.fund_eoa()
    gas_costs = fork.gas_costs()

    delegation_target = pre.deploy_contract(code=Op.STOP)
    target = pre.fund_eoa(amount=0, delegation=delegation_target)

    # memory expansion / no expansion
    ret_size = 32 if memory_expansion else 0
    ret_offset = 0

    # Use gas=1 to test that when we have enough for delegation_cost but not
    # enough for delegation_cost + message_gas, we still don't read the
    # delegation (the EVM does a static check upfront)
    message_gas = 1
    staticcall_code = Op.STATICCALL(
        address=target,
        gas=message_gas,
        ret_size=ret_size,
        ret_offset=ret_offset,
    )

    caller = pre.deploy_contract(code=staticcall_code)

    # Build access list for warming
    access_list: list[AccessList] = []
    if target_is_warm:
        access_list.append(AccessList(address=target, storage_keys=[]))
    if delegation_is_warm:
        access_list.append(
            AccessList(address=delegation_target, storage_keys=[])
        )
    access_list_or_none = access_list if access_list else None

    intrinsic_cost = fork.transaction_intrinsic_cost_calculator()(
        access_list=access_list_or_none
    )

    bytecode_cost = gas_costs.G_VERY_LOW * 6

    access_cost = (
        gas_costs.G_WARM_ACCOUNT_ACCESS
        if target_is_warm
        else gas_costs.G_COLD_ACCOUNT_ACCESS
    )
    memory_cost = fork.memory_expansion_gas_calculator()(new_bytes=ret_size)
    delegation_cost = (
        gas_costs.G_WARM_ACCOUNT_ACCESS
        if delegation_is_warm
        else gas_costs.G_COLD_ACCOUNT_ACCESS
    )

    static_gas_cost = access_cost + memory_cost
    # The EVM's second check cost is static_gas + delegation_cost.
    # When gas_left == extra_gas, calculate_message_call_gas caps the
    # provided gas to 0, so message_gas doesn't contribute to the cost.
    second_check_cost = static_gas_cost + delegation_cost

    if oog_boundary == OutOfGasBoundary.OOG_BEFORE_TARGET_ACCESS:
        gas_limit = intrinsic_cost + bytecode_cost + static_gas_cost - 1
    elif oog_boundary == OutOfGasBoundary.OOG_AFTER_TARGET_ACCESS:
        # Enough for static_gas only - not enough for delegation_cost.
        gas_limit = intrinsic_cost + bytecode_cost + static_gas_cost
    elif oog_boundary == OutOfGasBoundary.OOG_SUCCESS_MINUS_1:
        # One less than second_check_cost. This hits the early return path
        # in calculate_message_call_gas where message_gas IS added to cost,
        # causing OOG. Proves the exact boundary of the second check.
        gas_limit = intrinsic_cost + bytecode_cost + second_check_cost - 1
    else:
        gas_limit = intrinsic_cost + bytecode_cost + second_check_cost

    tx = Transaction(
        sender=alice,
        to=caller,
        gas_limit=gas_limit,
        access_list=access_list_or_none,
    )

    # Access list warming does NOT add to BAL - only EVM execution does
    if oog_boundary == OutOfGasBoundary.OOG_BEFORE_TARGET_ACCESS:
        target_in_bal = False
        delegation_in_bal = False
    elif oog_boundary in (
        OutOfGasBoundary.OOG_AFTER_TARGET_ACCESS,
        OutOfGasBoundary.OOG_SUCCESS_MINUS_1,
    ):
        # Both cases: target accessed but not enough gas for full call
        # so delegation is NOT read (static check optimization)
        target_in_bal = True
        delegation_in_bal = False
    else:
        target_in_bal = True
        delegation_in_bal = True

    account_expectations: Dict[Address, BalAccountExpectation | None] = {
        caller: BalAccountExpectation.empty(),
        delegation_target: (
            BalAccountExpectation.empty() if delegation_in_bal else None
        ),
    }

    if target_in_bal:
        account_expectations[target] = BalAccountExpectation.empty()
    else:
        account_expectations[target] = None

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[tx],
                expected_block_access_list=BlockAccessListExpectation(
                    account_expectations=account_expectations
                ),
            )
        ],
        post={alice: Account(nonce=1)},
    )


@pytest.mark.parametrize(
    "oog_scenario,memory_offset,copy_size",
    [
        pytest.param("success", 0, 0, id="successful_extcodecopy"),
        pytest.param("oog_at_cold_access", 0, 0, id="oog_at_cold_access"),
        pytest.param(
            "oog_at_memory_large_offset",
            0x10000,
            32,
            id="oog_at_memory_large_offset",
        ),
        pytest.param(
            "oog_at_memory_boundary",
            256,
            32,
            id="oog_at_memory_boundary",
        ),
    ],
)
def test_bal_extcodecopy_and_oog(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
    fork: Fork,
    oog_scenario: str,
    memory_offset: int,
    copy_size: int,
) -> None:
    """
    Ensure BAL handles EXTCODECOPY and OOG during EXTCODECOPY appropriately.

    Tests various OOG scenarios:
    - success: EXTCODECOPY completes, target appears in BAL
    - oog_at_cold_access: OOG before cold access, target NOT in BAL
    - oog_at_memory_large_offset: OOG at memory expansion (large offset),
      target NOT in BAL
    - oog_at_memory_boundary: OOG at memory expansion (boundary case),
      target NOT in BAL

    Gas for all components (cold access + copy + memory expansion) must be
    checked BEFORE recording account access.
    """
    alice = pre.fund_eoa()
    gas_costs = fork.gas_costs()

    # Create target contract with some code
    target_contract = pre.deploy_contract(
        code=Bytecode(Op.PUSH1(0x42) + Op.STOP)
    )

    # Build EXTCODECOPY contract with appropriate PUSH sizes
    if memory_offset <= 0xFF:
        dest_push = Op.PUSH1(memory_offset)
    elif memory_offset <= 0xFFFF:
        dest_push = Op.PUSH2(memory_offset)
    else:
        dest_push = Op.PUSH3(memory_offset)

    extcodecopy_contract_code = Bytecode(
        Op.PUSH1(copy_size)
        + Op.PUSH1(0)  # codeOffset
        + dest_push  # destOffset
        + Op.PUSH20(target_contract)
        + Op.EXTCODECOPY
        + Op.STOP
    )

    extcodecopy_contract = pre.deploy_contract(code=extcodecopy_contract_code)

    intrinsic_gas_calculator = fork.transaction_intrinsic_cost_calculator()
    intrinsic_gas_cost = intrinsic_gas_calculator()

    # Calculate costs
    push_cost = gas_costs.G_VERY_LOW * 4
    cold_access_cost = gas_costs.G_COLD_ACCOUNT_ACCESS
    copy_cost = gas_costs.G_COPY * ((copy_size + 31) // 32)

    if oog_scenario == "success":
        # Provide enough gas for everything including memory expansion
        memory_cost = fork.memory_expansion_gas_calculator()(
            new_bytes=memory_offset + copy_size
        )
        execution_cost = push_cost + cold_access_cost + copy_cost + memory_cost
        tx_gas_limit = intrinsic_gas_cost + execution_cost
        target_in_bal = True
    elif oog_scenario == "oog_at_cold_access":
        # Provide gas for pushes but 1 less than cold access cost
        execution_cost = push_cost + cold_access_cost
        tx_gas_limit = intrinsic_gas_cost + execution_cost - 1
        target_in_bal = False
    elif oog_scenario == "oog_at_memory_large_offset":
        # Provide gas for push + cold access + copy, but NOT memory expansion
        execution_cost = push_cost + cold_access_cost + copy_cost
        tx_gas_limit = intrinsic_gas_cost + execution_cost
        target_in_bal = False
    elif oog_scenario == "oog_at_memory_boundary":
        # Calculate memory cost and provide exactly 1 less than needed
        memory_cost = fork.memory_expansion_gas_calculator()(
            new_bytes=memory_offset + copy_size
        )
        execution_cost = push_cost + cold_access_cost + copy_cost + memory_cost
        tx_gas_limit = intrinsic_gas_cost + execution_cost - 1
        target_in_bal = False
    else:
        raise ValueError(f"Invariant: unknown oog_scenario {oog_scenario}")

    tx = Transaction(
        sender=alice,
        to=extcodecopy_contract,
        gas_limit=tx_gas_limit,
    )

    block = Block(
        txs=[tx],
        expected_block_access_list=BlockAccessListExpectation(
            account_expectations={
                extcodecopy_contract: BalAccountExpectation.empty(),
                **(
                    {target_contract: BalAccountExpectation.empty()}
                    if target_in_bal
                    else {target_contract: None}
                ),
            }
        ),
    )

    blockchain_test(
        pre=pre,
        blocks=[block],
        post={
            alice: Account(nonce=1),
            extcodecopy_contract: Account(),
            target_contract: Account(),
        },
    )


@pytest.mark.parametrize(
    "self_destruct_in_same_tx", [True, False], ids=["same_tx", "new_tx"]
)
@pytest.mark.parametrize(
    "pre_funded", [True, False], ids=["pre_funded", "not_pre_funded"]
)
def test_bal_self_destruct(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
    self_destruct_in_same_tx: bool,
    pre_funded: bool,
) -> None:
    """Ensure BAL captures balance changes caused by `SELFDESTRUCT`."""
    alice = pre.fund_eoa()
    bob = pre.fund_eoa(amount=0)

    selfdestruct_code = (
        Op.SLOAD(0x01)  # Read from storage slot 0x01
        + Op.SSTORE(0x02, 0x42)  # Write to storage slot 0x02
        + Op.SELFDESTRUCT(bob)
    )
    # A pre existing self-destruct contract with initial storage
    kaboom = pre.deploy_contract(code=selfdestruct_code, storage={0x01: 0x123})

    # A template for self-destruct contract
    self_destruct_init_code = Initcode(deploy_code=selfdestruct_code)
    template = pre.deploy_contract(code=self_destruct_init_code)

    transfer_amount = expected_recipient_balance = 100
    pre_fund_amount = 10

    if self_destruct_in_same_tx:
        # The goal is to create a self-destructing contract in the same
        # transaction to trigger deletion of code as per EIP-6780.
        # The factory contract below creates a new self-destructing
        # contract and calls it in this transaction.

        bytecode_size = len(self_destruct_init_code)
        factory_bytecode = (
            # Clone template memory
            Op.EXTCODECOPY(template, 0, 0, bytecode_size)
            # Fund 100 wei and deploy the clone
            + Op.CREATE(transfer_amount, 0, bytecode_size)
            # Call the clone, which self-destructs
            + Op.CALL(1_000_000, Op.DUP6, 0, 0, 0, 0, 0)
            + Op.STOP
        )

        factory = pre.deploy_contract(code=factory_bytecode)
        kaboom_same_tx = compute_create_address(address=factory, nonce=1)

    # Determine which account will be self-destructed
    self_destructed_account = (
        kaboom_same_tx if self_destruct_in_same_tx else kaboom
    )

    if pre_funded:
        expected_recipient_balance += pre_fund_amount
        pre.fund_address(
            address=self_destructed_account, amount=pre_fund_amount
        )

    tx = Transaction(
        sender=alice,
        to=factory if self_destruct_in_same_tx else kaboom,
        value=transfer_amount,
        gas_limit=1_000_000,
    )

    block = Block(
        txs=[tx],
        expected_block_access_list=BlockAccessListExpectation(
            account_expectations={
                alice: BalAccountExpectation(
                    nonce_changes=[BalNonceChange(tx_index=1, post_nonce=1)],
                ),
                bob: BalAccountExpectation(
                    balance_changes=[
                        BalBalanceChange(
                            tx_index=1, post_balance=expected_recipient_balance
                        )
                    ]
                ),
                self_destructed_account: BalAccountExpectation(
                    balance_changes=[
                        BalBalanceChange(tx_index=1, post_balance=0)
                    ]
                    if pre_funded
                    else [],
                    # Accessed slots for same-tx are recorded as reads (0x02)
                    storage_reads=[0x01, 0x02]
                    if self_destruct_in_same_tx
                    else [0x01],
                    # Storage changes are recorded for non-same-tx
                    # self-destructs
                    storage_changes=[
                        BalStorageSlot(
                            slot=0x02,
                            slot_changes=[
                                BalStorageChange(tx_index=1, post_value=0x42)
                            ],
                        )
                    ]
                    if not self_destruct_in_same_tx
                    else [],
                    code_changes=[],  # should not be present
                    nonce_changes=[],  # should not be present
                ),
            }
        ),
    )

    post: Dict[Address, Account] = {
        alice: Account(nonce=1),
        bob: Account(balance=expected_recipient_balance),
    }

    # If the account was self-destructed in the same transaction,
    # we expect the account to non-existent and its balance to be 0.
    if self_destruct_in_same_tx:
        post.update(
            {
                factory: Account(
                    nonce=2,  # incremented after CREATE
                    balance=0,  # spent on CREATE
                    code=factory_bytecode,
                ),
                kaboom_same_tx: Account.NONEXISTENT,  # type: ignore
                # The pre-existing contract remains unaffected
                kaboom: Account(
                    balance=0, code=selfdestruct_code, storage={0x01: 0x123}
                ),
            }
        )
    else:
        post.update(
            {
                # This contract was self-destructed in a separate tx.
                # From EIP 6780: `SELFDESTRUCT` does not delete any data
                # (including storage keys, code, or the account itself).
                kaboom: Account(
                    balance=0,
                    code=selfdestruct_code,
                    storage={0x01: 0x123, 0x2: 0x42},
                ),
            }
        )

    blockchain_test(
        pre=pre,
        blocks=[block],
        post=post,
    )


@pytest.mark.parametrize("oog_before_state_access", [True, False])
def test_bal_self_destruct_oog(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
    fork: Fork,
    oog_before_state_access: bool,
) -> None:
    """
    Test SELFDESTRUCT BAL behavior at gas boundaries.

    SELFDESTRUCT has two gas checkpoints:
    1. static checks: G_SELF_DESTRUCT + G_COLD_ACCOUNT_ACCESS
       OOG here = no state access, beneficiary NOT in BAL
    2. state access: same as static checks, plus G_NEW_ACCOUNT for new account
       OOG here = enough gas to access state but not enough for new account,
       beneficiary IS in BAL
    """
    alice = pre.fund_eoa()
    # always use new account so we incur extra G_NEW_ACCOUNT cost
    # there is no other gas boundary to test between cold access
    # and new account
    beneficiary = pre.empty_account()

    # selfdestruct_contract: PUSH20 <beneficiary> SELFDESTRUCT
    selfdestruct_code = Op.SELFDESTRUCT(beneficiary)
    selfdestruct_contract = pre.deploy_contract(
        code=selfdestruct_code, balance=1000
    )

    # Gas needed inside the CALL for SELFDESTRUCT:
    # - PUSH20: G_VERY_LOW = 3
    # - SELFDESTRUCT: G_SELF_DESTRUCT
    # - G_COLD_ACCOUNT_ACCESS (beneficiary cold access)
    gas_costs = fork.gas_costs()
    exact_static_gas = (
        gas_costs.G_VERY_LOW
        + gas_costs.G_SELF_DESTRUCT
        + gas_costs.G_COLD_ACCOUNT_ACCESS
    )

    # subtract one from the exact gas to trigger OOG before state access
    oog_gas = (
        exact_static_gas - 1 if oog_before_state_access else exact_static_gas
    )

    # caller_contract: CALL with oog_gas
    caller_code = Op.CALL(gas=oog_gas, address=selfdestruct_contract)
    caller_contract = pre.deploy_contract(code=caller_code)

    tx = Transaction(
        sender=alice,
        to=caller_contract,
        gas_limit=100_000,
    )

    account_expectations: Dict[Address, BalAccountExpectation | None] = {
        alice: BalAccountExpectation(
            nonce_changes=[BalNonceChange(tx_index=1, post_nonce=1)],
        ),
        caller_contract: BalAccountExpectation.empty(),
        selfdestruct_contract: BalAccountExpectation.empty(),
        # beneficiary only in BAL if we passed check_gas (state accessed)
        beneficiary: None
        if oog_before_state_access
        else BalAccountExpectation.empty(),
    }

    block = Block(
        txs=[tx],
        expected_block_access_list=BlockAccessListExpectation(
            account_expectations=account_expectations
        ),
    )

    blockchain_test(
        pre=pre,
        blocks=[block],
        post={
            alice: Account(nonce=1),
            caller_contract: Account(code=caller_code),
            # selfdestruct_contract still exists - SELFDESTRUCT failed
            selfdestruct_contract: Account(
                balance=1000, code=selfdestruct_code
            ),
        },
    )


def test_bal_storage_write_read_same_frame(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Ensure BAL captures write precedence over read in same call frame.

    Oracle writes to slot 0x01, then reads from slot 0x01 in same call.
    The write shadows the read - only the write appears in BAL.
    """
    alice = pre.fund_eoa()

    oracle_code = (
        Op.SSTORE(0x01, 0x42)  # Write 0x42 to slot 0x01
        + Op.SLOAD(0x01)  # Read from slot 0x01
        + Op.STOP
    )
    oracle = pre.deploy_contract(code=oracle_code, storage={0x01: 0x99})

    tx = Transaction(
        sender=alice,
        to=oracle,
        gas_limit=1_000_000,
    )

    block = Block(
        txs=[tx],
        expected_block_access_list=BlockAccessListExpectation(
            account_expectations={
                alice: BalAccountExpectation(
                    nonce_changes=[BalNonceChange(tx_index=1, post_nonce=1)],
                ),
                oracle: BalAccountExpectation(
                    storage_changes=[
                        BalStorageSlot(
                            slot=0x01,
                            slot_changes=[
                                BalStorageChange(tx_index=1, post_value=0x42)
                            ],
                        )
                    ],
                    storage_reads=[],  # Empty! Write shadows the read
                ),
            }
        ),
    )

    blockchain_test(
        pre=pre,
        blocks=[block],
        post={
            alice: Account(nonce=1),
            oracle: Account(storage={0x01: 0x42}),
        },
    )


@pytest.mark.parametrize(
    "call_opcode",
    [
        pytest.param(
            lambda target: Op.CALL(100_000, target, 0, 0, 0, 0, 0), id="call"
        ),
        pytest.param(
            lambda target: Op.DELEGATECALL(100_000, target, 0, 0, 0, 0),
            id="delegatecall",
        ),
        pytest.param(
            lambda target: Op.CALLCODE(100_000, target, 0, 0, 0, 0, 0),
            id="callcode",
        ),
    ],
)
def test_bal_storage_write_read_cross_frame(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
    call_opcode: Callable[[Bytecode], Bytecode],
) -> None:
    """
    Ensure BAL captures write precedence over read across call frames.

    Frame 1: Read slot 0x01 (0x99), write 0x42, then call itself.
    Frame 2: Read slot 0x01 (0x42), see it's 0x42 and return.
    Both reads are shadowed by the write - only write appears in BAL.
    """
    alice = pre.fund_eoa()

    # Oracle code:
    # 1. Read slot 0x01 (initial: 0x99, recursive: 0x42)
    # 2. If value == 0x42, return (exit recursion)
    # 3. Write 0x42 to slot 0x01
    # 4. Call itself recursively
    oracle_code = (
        Op.SLOAD(0x01)  # Load value from slot 0x01
        + Op.PUSH1(0x42)  # Push 0x42 for comparison
        + Op.EQ  # Check if loaded value == 0x42
        + Op.PUSH1(0x1D)  # Jump destination (after SSTORE + CALL)
        + Op.JUMPI  # If equal, jump to end (exit recursion)
        + Op.PUSH1(0x42)  # Value to write
        + Op.PUSH1(0x01)  # Slot 0x01
        + Op.SSTORE  # Write 0x42 to slot 0x01
        + call_opcode(Op.ADDRESS)  # Call itself
        + Op.JUMPDEST  # Jump destination for exit
        + Op.STOP
    )

    oracle = pre.deploy_contract(code=oracle_code, storage={0x01: 0x99})

    tx = Transaction(
        sender=alice,
        to=oracle,
        gas_limit=1_000_000,
    )

    block = Block(
        txs=[tx],
        expected_block_access_list=BlockAccessListExpectation(
            account_expectations={
                alice: BalAccountExpectation(
                    nonce_changes=[BalNonceChange(tx_index=1, post_nonce=1)],
                ),
                oracle: BalAccountExpectation(
                    storage_changes=[
                        BalStorageSlot(
                            slot=0x01,
                            slot_changes=[
                                BalStorageChange(tx_index=1, post_value=0x42)
                            ],
                        )
                    ],
                    storage_reads=[],  # Empty! Write shadows both reads
                ),
            }
        ),
    )

    blockchain_test(
        pre=pre,
        blocks=[block],
        post={
            alice: Account(nonce=1),
            oracle: Account(storage={0x01: 0x42}),
        },
    )


def test_bal_create_oog_code_deposit(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
    fork: Fork,
) -> None:
    """
    Ensure BAL correctly handles CREATE that runs out of gas during code
    deposit. The contract address should appear with empty changes (read
    during collision check) but no nonce or code changes (rolled back).
    """
    alice = pre.fund_eoa()

    # create init code that returns a very large contract to force OOG
    deposited_len = 10_000
    initcode = Op.RETURN(0, deposited_len)

    factory = pre.deploy_contract(
        code=Op.MSTORE(0, Op.PUSH32(bytes(initcode)))
        + Op.SSTORE(
            1, Op.CREATE(offset=32 - len(initcode), size=len(initcode))
        )
        + Op.STOP,
        storage={1: 0xDEADBEEF},
    )

    contract_address = compute_create_address(address=factory, nonce=1)

    intrinsic_gas_calculator = fork.transaction_intrinsic_cost_calculator()
    intrinsic_gas = intrinsic_gas_calculator(
        calldata=b"",
        contract_creation=False,
        access_list=[],
    )

    tx = Transaction(
        sender=alice,
        to=factory,
        gas_limit=intrinsic_gas + 500_000,  # insufficient for deposit
    )

    # BAL expectations:
    # - Alice: nonce change (tx sender)
    # - Factory: nonce change (CREATE increments factory nonce)
    # - Contract address: empty changes (read during collision check,
    #   nonce/code changes rolled back on OOG)
    account_expectations = {
        alice: BalAccountExpectation(
            nonce_changes=[BalNonceChange(tx_index=1, post_nonce=1)],
        ),
        factory: BalAccountExpectation(
            nonce_changes=[BalNonceChange(tx_index=1, post_nonce=2)],
            storage_changes=[
                BalStorageSlot(
                    slot=1,
                    slot_changes=[
                        # SSTORE saves 0 (CREATE failed)
                        BalStorageChange(tx_index=1, post_value=0),
                    ],
                )
            ],
        ),
        contract_address: BalAccountExpectation.empty(),
    }

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[tx],
                expected_block_access_list=BlockAccessListExpectation(
                    account_expectations=account_expectations
                ),
            )
        ],
        post={
            alice: Account(nonce=1),
            factory: Account(nonce=2, storage={1: 0}),
            contract_address: Account.NONEXISTENT,
        },
    )


def test_bal_sstore_static_context(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Ensure BAL does not record storage reads when SSTORE fails in static
    context.

    Contract A makes STATICCALL to Contract B. Contract B attempts SSTORE,
    which should fail immediately without recording any storage reads.
    """
    alice = pre.fund_eoa()

    contract_b = pre.deploy_contract(code=Op.SSTORE(0, 5))

    # Contract A makes STATICCALL to Contract B
    # The STATICCALL will fail because B tries SSTORE in static context
    # But contract_a continues and writes to its own storage
    contract_a = pre.deploy_contract(
        code=Op.STATICCALL(
            gas=1_000_000,
            address=contract_b,
            args_offset=0,
            args_size=0,
            ret_offset=0,
            ret_size=0,
        )
        + Op.POP  # pop the return value (0 = failure)
        + Op.SSTORE(0, 1)  # this should succeed (non-static context)
    )

    tx = Transaction(
        sender=alice,
        to=contract_a,
        gas_limit=2_000_000,
    )

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[tx],
                expected_block_access_list=BlockAccessListExpectation(
                    account_expectations={
                        alice: BalAccountExpectation(
                            nonce_changes=[
                                BalNonceChange(tx_index=1, post_nonce=1)
                            ],
                        ),
                        contract_a: BalAccountExpectation(
                            storage_changes=[
                                BalStorageSlot(
                                    slot=0x00,
                                    slot_changes=[
                                        BalStorageChange(
                                            tx_index=1, post_value=1
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        contract_b: BalAccountExpectation.empty(),
                    }
                ),
            )
        ],
        post={
            contract_a: Account(storage={0: 1}),
            contract_b: Account(storage={0: 0}),  # SSTORE failed
        },
    )


def test_bal_create_contract_init_revert(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
) -> None:
    """
    Test that BAL does not include nonce/code changes when CREATE happens
    in a call that then REVERTs.
    """
    alice = pre.fund_eoa(amount=10**18)

    # Simple init code that returns STOP as deployed code
    init_code_bytes = bytes(Op.RETURN(0, 1) + Op.STOP)

    # Factory that does CREATE then REVERTs
    factory = pre.deploy_contract(
        code=Op.MSTORE(0, Op.PUSH32(init_code_bytes))
        + Op.POP(Op.CREATE(0, 32 - len(init_code_bytes), len(init_code_bytes)))
        + Op.REVERT(0, 0)
    )

    # A caller that CALLs factory to CREATE then REVERT
    caller = pre.deploy_contract(code=Op.CALL(address=factory))

    created_address = compute_create_address(address=factory, nonce=1)

    tx = Transaction(
        sender=alice,
        to=caller,
        gas_limit=500_000,
    )

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=[tx],
                expected_block_access_list=BlockAccessListExpectation(
                    account_expectations={
                        alice: BalAccountExpectation(
                            nonce_changes=[
                                BalNonceChange(tx_index=1, post_nonce=1)
                            ],
                        ),
                        caller: BalAccountExpectation.empty(),
                        factory: BalAccountExpectation.empty(),
                        created_address: BalAccountExpectation.empty(),
                    }
                ),
            )
        ],
        post={
            alice: Account(nonce=1),
            caller: Account(nonce=1),
            factory: Account(nonce=1),
            created_address: Account.NONEXISTENT,
        },
    )


def test_bal_call_revert_insufficient_funds(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Test BAL with CALL failure due to insufficient balance (not OOG).

    Contract (balance=100): SLOAD(0x01)→CALL(target, value=1000)→SSTORE(0x02).
    CALL fails because 1000 > 100. Target is 0xDEAD.

    Expected BAL:
    - Contract: storage_reads [0x01], storage_changes slot 0x02 (value=0)
    - Target: appears in BAL (accessed before balance check fails)
    """
    alice = pre.fund_eoa()

    contract_balance = 100
    transfer_amount = 1000  # More than contract has

    # Target address that should be warmed but not receive funds
    # Give it a small balance so it's not considered "empty" and pruned
    target_balance = 1
    target_address = pre.fund_eoa(amount=target_balance)

    # Contract that:
    # 1. SLOAD slot 0x01
    # 2. CALL target with value=1000 (will fail - insufficient funds)
    # 3. SSTORE slot 0x02 with CALL result (0 = failure)
    contract_code = (
        Op.SLOAD(0x01)  # Read from slot 0x01, push to stack
        + Op.POP  # Discard value
        # CALL(gas, addr, value, argsOffset, argsSize, retOffset, retSize)
        + Op.CALL(100_000, target_address, transfer_amount, 0, 0, 0, 0)
        # CALL result is on stack (0 = failure, 1 = success)
        # Stack: [result]
        + Op.PUSH1(0x02)  # Push slot number
        # Stack: [0x02, result]
        + Op.SSTORE  # SSTORE pops slot (0x02), then value (result)
        + Op.STOP
    )

    contract = pre.deploy_contract(
        code=contract_code,
        balance=contract_balance,
        storage={
            0x02: 0xDEAD
        },  # Non-zero initial value so SSTORE(0) is a change
    )

    tx = Transaction(
        sender=alice,
        to=contract,
        gas_limit=1_000_000,
    )

    block = Block(
        txs=[tx],
        expected_block_access_list=BlockAccessListExpectation(
            account_expectations={
                alice: BalAccountExpectation(
                    nonce_changes=[BalNonceChange(tx_index=1, post_nonce=1)],
                ),
                contract: BalAccountExpectation(
                    # Storage read for slot 0x01
                    storage_reads=[0x01],
                    # Storage change for slot 0x02 (CALL result = 0)
                    storage_changes=[
                        BalStorageSlot(
                            slot=0x02,
                            slot_changes=[
                                BalStorageChange(tx_index=1, post_value=0)
                            ],
                        )
                    ],
                ),
                # Target appears in BAL - accessed before balance check fails
                target_address: BalAccountExpectation.empty(),
            }
        ),
    )

    blockchain_test(
        pre=pre,
        blocks=[block],
        post={
            alice: Account(nonce=1),
            contract: Account(
                balance=contract_balance,  # Unchanged - transfer failed
                storage={0x02: 0},  # CALL returned 0 (failure)
            ),
            target_address: Account(balance=target_balance),  # Unchanged
        },
    )


def test_bal_create_selfdestruct_to_self_with_call(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Test BAL with init code that CALLs Oracle, writes storage, then
    SELFDESTRUCTs to self.

    Factory CREATE2(endowment=100).
    Init: CALL(Oracle)→SSTORE(0x01)→SELFDESTRUCT(SELF).

    Expected BAL:
    - Factory: nonce_changes, balance_changes (loses 100)
    - Oracle: storage_changes slot 0x01
    - Created address: storage_reads [0x01] (aborted write→read),
      MUST NOT have nonce/code/storage/balance changes (ephemeral)
    """
    alice = pre.fund_eoa()
    factory_balance = 1000

    # Oracle contract that writes to slot 0x01 when called
    oracle_code = Op.SSTORE(0x01, 0x42) + Op.STOP
    oracle = pre.deploy_contract(code=oracle_code)

    endowment = 100

    # Init code that:
    # 1. Calls Oracle (which writes to its slot 0x01)
    # 2. Writes 0x42 to own slot 0x01
    # 3. Selfdestructs to self
    initcode_runtime = (
        # CALL(gas, Oracle, value=0, ...)
        Op.CALL(100_000, oracle, 0, 0, 0, 0, 0)
        + Op.POP
        # Write to own storage slot 0x01
        + Op.SSTORE(0x01, 0x42)
        # SELFDESTRUCT to self (ADDRESS returns own address)
        + Op.SELFDESTRUCT(Op.ADDRESS)
    )
    init_code = Initcode(deploy_code=Op.STOP, initcode_prefix=initcode_runtime)
    init_code_bytes = bytes(init_code)
    init_code_size = len(init_code_bytes)

    # Factory code with embedded initcode (no template contract needed)
    # Structure: [execution code] [initcode bytes]
    # CODECOPY copies initcode from factory's own code to memory
    #
    # Two-pass approach: build with placeholder, measure, rebuild
    placeholder_offset = 0xFF  # Placeholder (same byte size as final value)
    factory_execution_template = (
        Op.CODECOPY(0, placeholder_offset, init_code_size)
        + Op.SSTORE(
            0x00,
            Op.CREATE2(
                value=endowment,
                offset=0,
                size=init_code_size,
                salt=0,
            ),
        )
        + Op.STOP
    )
    # Measure execution code size
    execution_code_size = len(bytes(factory_execution_template))

    # Rebuild with actual offset value
    factory_execution = (
        Op.CODECOPY(0, execution_code_size, init_code_size)
        + Op.SSTORE(
            0x00,
            Op.CREATE2(
                value=endowment,
                offset=0,
                size=init_code_size,
                salt=0,
            ),
        )
        + Op.STOP
    )
    # Combine execution code with embedded initcode
    factory_code = bytes(factory_execution) + init_code_bytes

    factory = pre.deploy_contract(code=factory_code, balance=factory_balance)

    # Calculate the CREATE2 target address
    created_address = compute_create_address(
        address=factory,
        nonce=1,
        salt=0,
        initcode=init_code_bytes,
        opcode=Op.CREATE2,
    )

    tx = Transaction(
        sender=alice,
        to=factory,
        gas_limit=1_000_000,
    )

    block = Block(
        txs=[tx],
        expected_block_access_list=BlockAccessListExpectation(
            account_expectations={
                alice: BalAccountExpectation(
                    nonce_changes=[BalNonceChange(tx_index=1, post_nonce=1)],
                ),
                factory: BalAccountExpectation(
                    nonce_changes=[BalNonceChange(tx_index=1, post_nonce=2)],
                    # Balance changes: loses endowment (100)
                    balance_changes=[
                        BalBalanceChange(
                            tx_index=1,
                            post_balance=factory_balance - endowment,
                        )
                    ],
                ),
                # Oracle: storage changes for slot 0x01
                oracle: BalAccountExpectation(
                    storage_changes=[
                        BalStorageSlot(
                            slot=0x01,
                            slot_changes=[
                                BalStorageChange(tx_index=1, post_value=0x42)
                            ],
                        )
                    ],
                ),
                # Created address: ephemeral (created and destroyed same tx)
                # - storage_reads for slot 0x01 (aborted write becomes read)
                # - NO nonce/code/storage/balance changes
                created_address: BalAccountExpectation(
                    storage_reads=[0x01],
                    storage_changes=[],
                    nonce_changes=[],
                    code_changes=[],
                    balance_changes=[],
                ),
            }
        ),
    )

    blockchain_test(
        pre=pre,
        blocks=[block],
        post={
            alice: Account(nonce=1),
            factory: Account(nonce=2, balance=factory_balance - endowment),
            oracle: Account(storage={0x01: 0x42}),
            # Created address doesn't exist - destroyed in same tx
            created_address: Account.NONEXISTENT,
        },
    )


def test_bal_create2_collision(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Test BAL with CREATE2 collision against pre-existing contract.

    Pre-existing contract has code=STOP, nonce=1.
    Factory (nonce=1, slot[0]=0xDEAD) executes CREATE2 targeting it.

    Expected BAL:
    - Factory: nonce_changes (1→2), storage_changes slot 0 (0xDEAD→0)
    - Collision address: empty (accessed during collision check)
    - Collision address MUST NOT have nonce_changes or code_changes
    """
    alice = pre.fund_eoa()

    # Init code that deploys simple STOP contract
    init_code = Initcode(deploy_code=Op.STOP)
    init_code_bytes = bytes(init_code)

    # Factory code: CREATE2 and store result in slot 0
    factory_code = (
        # Push init code to memory
        Op.MSTORE(0, Op.PUSH32(init_code_bytes))
        # SSTORE(0, CREATE2(...)) - stores CREATE2 result in slot 0
        + Op.SSTORE(
            0x00,
            Op.CREATE2(
                value=0,
                offset=32 - len(init_code_bytes),
                size=len(init_code_bytes),
                salt=0,
            ),
        )
        + Op.STOP
    )

    # Deploy factory - it starts with nonce=1 by default
    factory = pre.deploy_contract(
        code=factory_code,
        storage={0x00: 0xDEAD},  # Initial value to prove SSTORE works
    )

    # Calculate the CREATE2 target address
    collision_address = compute_create_address(
        address=factory,
        nonce=1,
        salt=0,
        initcode=init_code_bytes,
        opcode=Op.CREATE2,
    )

    # Set up the collision by pre-populating the target address
    # This contract has code (STOP) and nonce=1, causing collision
    pre[collision_address] = Account(
        code=Op.STOP,
        nonce=1,
    )

    tx = Transaction(
        sender=alice,
        to=factory,
        gas_limit=1_000_000,
    )

    block = Block(
        txs=[tx],
        expected_block_access_list=BlockAccessListExpectation(
            account_expectations={
                alice: BalAccountExpectation(
                    nonce_changes=[BalNonceChange(tx_index=1, post_nonce=1)],
                ),
                factory: BalAccountExpectation(
                    # Nonce incremented 1→2 even on failed CREATE2
                    nonce_changes=[BalNonceChange(tx_index=1, post_nonce=2)],
                    # Storage changes: slot 0 = 0xDEAD → 0 (CREATE2 returned 0)
                    storage_changes=[
                        BalStorageSlot(
                            slot=0x00,
                            slot_changes=[
                                BalStorageChange(tx_index=1, post_value=0)
                            ],
                        )
                    ],
                ),
                # Collision address: empty (accessed but no state changes)
                # Explicitly verify ALL fields are empty
                collision_address: BalAccountExpectation(
                    nonce_changes=[],  # MUST NOT have nonce changes
                    balance_changes=[],  # MUST NOT have balance changes
                    code_changes=[],  # MUST NOT have code changes
                    storage_changes=[],  # MUST NOT have storage changes
                    storage_reads=[],  # MUST NOT have storage reads
                ),
            }
        ),
    )

    blockchain_test(
        pre=pre,
        blocks=[block],
        post={
            alice: Account(nonce=1),
            factory: Account(nonce=2, storage={0x00: 0}),
            # Collision address unchanged - contract still exists
            collision_address: Account(code=bytes(Op.STOP), nonce=1),
        },
    )


def test_bal_transient_storage_not_tracked(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Test BAL excludes EIP-1153 transient storage (TSTORE/TLOAD).

    Contract: TSTORE(0x01, 0x42)→TLOAD(0x01)→SSTORE(0x02, result).

    Expected BAL:
    - storage_changes: slot 0x02 (persistent)
    - MUST NOT include slot 0x01 (transient storage not persisted)
    """
    alice = pre.fund_eoa()

    # Contract that uses transient storage then persists to regular storage
    contract_code = (
        # TSTORE slot 0x01 with value 0x42 (transient storage)
        Op.TSTORE(0x01, 0x42)
        # TLOAD slot 0x01 (transient storage read)
        + Op.TLOAD(0x01)
        # Result (0x42) is on stack, store it in persistent slot 0x02
        + Op.PUSH1(0x02)
        + Op.SSTORE  # SSTORE pops slot (0x02), then value (0x42)
        + Op.STOP
    )

    contract = pre.deploy_contract(code=contract_code)

    tx = Transaction(
        sender=alice,
        to=contract,
        gas_limit=1_000_000,
    )

    block = Block(
        txs=[tx],
        expected_block_access_list=BlockAccessListExpectation(
            account_expectations={
                alice: BalAccountExpectation(
                    nonce_changes=[BalNonceChange(tx_index=1, post_nonce=1)],
                ),
                contract: BalAccountExpectation(
                    # Persistent storage change for slot 0x02
                    storage_changes=[
                        BalStorageSlot(
                            slot=0x02,
                            slot_changes=[
                                BalStorageChange(tx_index=1, post_value=0x42)
                            ],
                        )
                    ],
                    # MUST NOT include slot 0x01 in storage_reads
                    # Transient storage operations don't pollute BAL
                    storage_reads=[],
                ),
            }
        ),
    )

    blockchain_test(
        pre=pre,
        blocks=[block],
        post={
            alice: Account(nonce=1),
            contract: Account(storage={0x02: 0x42}),
        },
    )


def test_bal_selfdestruct_to_precompile(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Test BAL with SELFDESTRUCT to precompile (ecrecover 0x01).

    Victim (balance=100) selfdestructs to precompile 0x01.

    Expected BAL:
    - Victim: balance_changes (100→0)
    - Precompile 0x01: balance_changes (0→100), no code/nonce changes
    """
    alice = pre.fund_eoa()

    contract_balance = 100
    ecrecover_precompile = Address(1)  # 0x0000...0001

    # Contract that selfdestructs to ecrecover precompile
    victim_code = Op.SELFDESTRUCT(ecrecover_precompile)

    victim = pre.deploy_contract(code=victim_code, balance=contract_balance)

    # Caller that triggers the selfdestruct
    caller_code = Op.CALL(100_000, victim, 0, 0, 0, 0, 0) + Op.STOP
    caller = pre.deploy_contract(code=caller_code)

    tx = Transaction(
        sender=alice,
        to=caller,
        gas_limit=1_000_000,
    )

    block = Block(
        txs=[tx],
        expected_block_access_list=BlockAccessListExpectation(
            account_expectations={
                alice: BalAccountExpectation(
                    nonce_changes=[BalNonceChange(tx_index=1, post_nonce=1)],
                ),
                caller: BalAccountExpectation.empty(),
                # Victim (selfdestructing contract): balance changes 100→0
                # Explicitly verify ALL fields to avoid false positives
                victim: BalAccountExpectation(
                    nonce_changes=[],  # Contract nonce unchanged
                    balance_changes=[
                        BalBalanceChange(tx_index=1, post_balance=0)
                    ],
                    code_changes=[],  # Code unchanged (post-Cancun)
                    storage_changes=[],  # No storage changes
                    storage_reads=[],  # No storage reads
                ),
                # Precompile receives selfdestruct balance
                # Explicitly verify ALL fields to avoid false positives
                ecrecover_precompile: BalAccountExpectation(
                    nonce_changes=[],  # MUST NOT have nonce changes
                    balance_changes=[
                        BalBalanceChange(
                            tx_index=1, post_balance=contract_balance
                        )
                    ],
                    code_changes=[],  # MUST NOT have code changes
                    storage_changes=[],  # MUST NOT have storage changes
                    storage_reads=[],  # MUST NOT have storage reads
                ),
            }
        ),
    )

    blockchain_test(
        pre=pre,
        blocks=[block],
        post={
            alice: Account(nonce=1),
            caller: Account(),
            # Victim still exists with 0 balance (post-Cancun SELFDESTRUCT)
            victim: Account(balance=0),
            # Precompile has received the balance
            ecrecover_precompile: Account(balance=contract_balance),
        },
    )


def test_bal_create_early_failure(
    pre: Alloc,
    blockchain_test: BlockchainTestFiller,
) -> None:
    """
    Test BAL with CREATE failure due to insufficient endowment.

    Factory (balance=50) attempts CREATE(value=100).
    Fails before nonce increment (before track_address).
    Distinct from collision where address IS accessed.

    Expected BAL:
    - Alice: nonce_changes
    - Factory: storage_changes slot 0 (0xDEAD→0), NO nonce_changes
    - Contract address: MUST NOT appear (never accessed)
    """
    alice = pre.fund_eoa()

    factory_balance = 50
    endowment = 100  # More than factory has

    # Simple init code that deploys STOP
    init_code = Initcode(deploy_code=Op.STOP)
    init_code_bytes = bytes(init_code)

    # Factory code: CREATE(value=endowment) and store result in slot 0
    factory_code = (
        # Push init code to memory
        Op.MSTORE(0, Op.PUSH32(init_code_bytes))
        # SSTORE(0, CREATE(value, offset, size))
        + Op.SSTORE(
            0x00,
            Op.CREATE(
                value=endowment,  # 100 > 50, will fail
                offset=32 - len(init_code_bytes),
                size=len(init_code_bytes),
            ),
        )
        + Op.STOP
    )

    # Deploy factory with insufficient balance for the CREATE endowment
    factory = pre.deploy_contract(
        code=factory_code,
        balance=factory_balance,
        storage={0x00: 0xDEAD},  # Initial value to prove SSTORE works
    )

    # Calculate what the contract address WOULD be (but it won't be created)
    would_be_contract_address = compute_create_address(
        address=factory, nonce=1
    )

    tx = Transaction(
        sender=alice,
        to=factory,
        gas_limit=1_000_000,
    )

    block = Block(
        txs=[tx],
        expected_block_access_list=BlockAccessListExpectation(
            account_expectations={
                alice: BalAccountExpectation(
                    nonce_changes=[BalNonceChange(tx_index=1, post_nonce=1)],
                ),
                factory: BalAccountExpectation(
                    # NO nonce_changes - CREATE failed before increment_nonce
                    nonce_changes=[],
                    # Storage changes: slot 0 = 0xDEAD → 0 (CREATE returned 0)
                    storage_changes=[
                        BalStorageSlot(
                            slot=0x00,
                            slot_changes=[
                                BalStorageChange(tx_index=1, post_value=0)
                            ],
                        )
                    ],
                ),
                # Contract address MUST NOT appear in BAL - never accessed
                # (CREATE failed before track_address was called)
                would_be_contract_address: None,
            }
        ),
    )

    blockchain_test(
        pre=pre,
        blocks=[block],
        post={
            alice: Account(nonce=1),
            # Factory nonce unchanged (still 1), balance unchanged
            factory: Account(
                nonce=1, balance=factory_balance, storage={0x00: 0}
            ),
            # Contract was never created
            would_be_contract_address: Account.NONEXISTENT,
        },
    )


@pytest.fixture(scope="module", autouse=True)
def print_aggregate_at_end():
    """Print aggregate gas tables at the end of test module."""
    yield  # Let all tests run
    print_aggregate_gas_table()
    print_delegatecall_no_delegation_aggregate_table()
