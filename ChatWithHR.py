#!/usr/bin/env python3
"""
Chat-with-Your-HR-Data — Enhanced Gender & Turnover Analysis
------------------------------------------------------------
Handles:
- Headcount (HC)
- Turnover rate
- Turnover comparison between years
- Gender split & gender comparison

With:
- Natural language answers
- Synonyms (HC, headcount, gender diversity, female share, etc.)
- Multiple filters (year + department + location + business unit)
- Clean chat (no JSON plan shown)
"""

import json
import re
import difflib
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import pandas as pd

# Ollama setup
try:
    import ollama
except ImportError:
    raise SystemExit("Install ollama first: pip install gradio pandas ollama openpyxl")

MODEL_NAME = "llama3:latest"
APP_TITLE = "Chat with Your HR Data"

# Aliases and heuristics
ALIASES = {
    "Calendar Year": ["calendar year", "year", "fiscal year", "calendar_year"],
    "Department": ["department", "dept", "organization", "org", "division"],
    "Employee ID": [
        "employee id", "employeeid", "employee_id", "emp id",
        "person number", "worker id", "person_number", "worker_id"
    ],
    "Employee Status": [
        "employee status", "status", "employment status", "termination status",
        "emp status", "emp_status"
    ],
    "Business Unit": [
        "business unit", "bu", "b.u.", "businessunit",
        "unit", "division", "team", "group", "business_unit"
    ],
    "Location": ["location", "site", "office", "country", "city"],
    "Gender": ["gender", "sex"],
}

GROUPABLE_PREFERENCES = ["Department", "Location", "Business Unit", "Gender"]
LIKELY_ID_COLUMNS = ["Employee ID", "employee_id", "Person Number", "Worker ID"]
SUPPORTED_OPERATORS = {"==", "!=", ">", ">=", "<", "<=", "in", "not in", "contains"}
SUPPORTED_METRICS = {
    "headcount_unique",
    "headcount_comparison",
    "turnover_rate",
    "turnover_comparison",
    "gender_share",
    "gender_comparison",
}

EXIT_STATUS_KEYWORDS = {
    "exit", "exited", "terminated", "termination",
    "separated", "separation", "left", "inactive",
    "voluntary termination", "involuntary termination"
}


def canonicalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def build_alias_map(df: pd.DataFrame) -> Dict[str, str]:
    alias_map = {}
    for col in df.columns:
        alias_map[canonicalize(col)] = col
        alias_map[col] = col
    for actual, synonyms in ALIASES.items():
        if actual in df.columns:
            for s in synonyms:
                alias_map[canonicalize(s)] = actual
                alias_map[s] = actual
    return alias_map


def resolve_column(token: str, alias_map: Dict[str, str], df: pd.DataFrame) -> Optional[str]:
    if token in df.columns:
        return token
    can = canonicalize(token)
    if can in alias_map:
        return alias_map[can]
    candidates_map = {canonicalize(c): c for c in df.columns}
    candidates = list(candidates_map.keys())
    match = difflib.get_close_matches(can, candidates, n=1, cutoff=0.85)
    if match:
        return candidates_map.get(match[0])
    return None


def pick_id_column(df: pd.DataFrame) -> Optional[str]:
    for col in LIKELY_ID_COLUMNS:
        if col in df.columns:
            return col
    for col in df.columns:
        try:
            if df[col].nunique(dropna=True) > 0.7 * len(df[col]):
                return col
        except Exception:
            continue
    return None


def _to_list_if_needed(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        parts = [v.strip() for v in val.split(",")]
        return [p for p in parts if p]
    return [val]


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# LLM planning

SYSTEM_INSTRUCTIONS = (
    "Convert user questions into STRICT JSON (no text, no markdown).\n"
    "Intent must be 'metric'.\n"
    "\n"
    "You MUST extract ALL filters mentioned in the user question.\n"
    "If the user specifies multiple conditions such as year + department + location + business unit,\n"
    "you MUST output multiple filter objects in the 'filters' list.\n"
    "\n"
    "Example 1 (year + department + location):\n"
    "User: 'total headcount in 2025 in Department 1 at location US'\n"
    "Then:\n"
    "  \"filters\": [\n"
    "    {\"column\": \"Calendar Year\", \"op\": \"==\", \"value\": 2025},\n"
    "    {\"column\": \"Department\", \"op\": \"==\", \"value\": \"Department 1\"},\n"
    "    {\"column\": \"Location\", \"op\": \"==\", \"value\": \"US\"}\n"
    "  ]\n"
    "\n"
    "Example 2 (year + business unit):\n"
    "User: 'headcount in Business Unit BU 12 in 2025'\n"
    "Then:\n"
    "  \"filters\": [\n"
    "    {\"column\": \"Business Unit\", \"op\": \"==\", \"value\": \"BU 12\"},\n"
    "    {\"column\": \"Calendar Year\", \"op\": \"==\", \"value\": 2025}\n"
    "  ]\n"
    "\n"
    "Supported metrics:\n"
    "  - 'headcount_unique' on the ID column to count unique employees.\n"
    "  - 'headcount_comparison' to compare headcount between two or more years and capture the change.\n"
    "  - 'turnover_rate' to compute unique exited employees divided by unique headcount\n"
    "    over the same slice (e.g., year/department/location/business unit).\n"
    "  - 'turnover_comparison' to compare turnover between two or more years and capture the change.\n"
    "  - 'gender_share' to compute the percentage (and optionally counts) of Male vs Female employees.\n"
    "  - 'gender_comparison' to compare gender distribution across multiple years.\n"
    "\n"
    "Synonyms for headcount (map to 'headcount_unique'):\n"
    "  headcount, hc, staff count, employee count, number of employees, fte count.\n"
    "\n"
    "Synonyms related to employees leaving (map to 'turnover_rate' or 'turnover_comparison'):\n"
    "  turnover, attrition, churn, people who left, leavers, exits,\n"
    "  turnover rate, attrition rate.\n"
    "If the user asks about change over time (e.g., 'how did turnover change between 2024 and 2025',\n"
    "'difference in turnover', 'turnover trend between years'), use metric.name = 'turnover_comparison'.\n"
    "\n"
    "For 'turnover_comparison', if the user mentions specific years, include them in a single filter\n"
    "using 'in', for example:\n"
    "  \"filters\": [\n"
    "    {\"column\": \"Calendar Year\", \"op\": \"in\", \"value\": [2024, 2025]}\n"
    "  ]\n"
    "You may also include additional filters such as Department, Location, Business Unit, etc.\n"
    "\n"
    "Synonyms related to gender distribution (map to 'gender_share'):\n"
    "  gender split, gender share, gender breakdown, gender diversity,\n"
    "  male female ratio, female share, gender balance, representation,\n"
    "  female representation, women representation, women share.\n"
    "\n"
    "Synonyms for Business Unit dimension:\n"
    "  Business Unit, BU, businessunit, unit, division, team, group.\n"
    "For the Business Unit dimension, ALWAYS use 'Business Unit' as the column name,\n"
    "and use values like 'BU 12' or 'Sales' as the 'value' field.\n"
    "\n"
    "Mapping rules:\n"
    "  - Phrases about employees leaving in one period → metric.name = 'turnover_rate'\n"
    "  - Phrases about change or comparison of turnover across years → 'turnover_comparison'\n"
    "  - Phrases about gender distribution → metric.name = 'gender_share'\n"
    "  - Phrases about comparing gender over time → metric.name = 'gender_comparison'\n"
    "  - Phrases about headcount / HC → metric.name = 'headcount_unique'\n"
    "  - Phrases about change or comparison of headcount / HC across years → metric.name = 'headcount_comparison'\n"
    "\n"
    "ALWAYS include these fields inside each metric object:\n"
    "  {\"name\", \"on\", \"alias\", \"show_count\", \"show_percentage\"}.\n"
    "The 'on' field should point to the ID column (e.g., 'Employee ID').\n"
    "\n"
    "Group-by rule:\n"
    "  - If user says 'by X', set group_by to [\"X\"].\n"
    "  - If user says 'by X and Y', set group_by to [\"X\", \"Y\"].\n"
    "  - Otherwise omit group_by.\n"
    "\n"
    "IMPORTANT: DO NOT add any filter on Employee Status unless explicitly stated.\n"
)

EXAMPLE_PLAN = {
    "version": "1.3",
    "intent": "metric",
    "filters": [
        {"column": "Calendar Year", "op": "==", "value": 2024}
    ],
    "group_by": ["Location"],
    "metrics": [
        {
            "name": "gender_share",
            "on": "Employee ID",
            "alias": "Gender Split",
            "show_percentage": True,
            "show_count": False,
        }
    ],
}


def make_planner_messages(question: str, df: pd.DataFrame) -> List[Dict[str, str]]:
    sys = (
        f"{SYSTEM_INSTRUCTIONS}\n"
        f"Columns in dataset: {list(df.columns)}\n"
        f"Example:\n{json.dumps(EXAMPLE_PLAN)}"
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": question},
    ]


def parse_plan(plan_text: str) -> Dict[str, Any]:
    try:
        return json.loads(plan_text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", plan_text)
        if m:
            return json.loads(m.group(0))
    raise ValueError("Model did not return valid JSON")


def normalize_question(question: str) -> str:
    """
    Normalize common synonyms so the model sees more consistent language.
    This does NOT change what is shown in the chat – only what we send to the LLM.
    """
    q = question

    replacements = {
        r"\bhc\b": " headcount ",
        r"\bhead count\b": " headcount ",
        r"\bstaff count\b": " headcount ",
        r"\bnumber of employees\b": " headcount ",
        r"\bemployee count\b": " headcount ",
        r"\bfte count\b": " headcount ",
        r"\bgender diversity\b": " gender share ",
        r"\bgender balance\b": " gender share ",
        r"\bfemale share\b": " gender share ",
        r"\bfemale representation\b": " gender share ",
        r"\bwomen representation\b": " gender share ",
        r"\bwomen share\b": " gender share ",
    }

    for pattern, repl in replacements.items():
        q = re.sub(pattern, repl, q, flags=re.IGNORECASE)

    return q


# Filters

def apply_filters(df: pd.DataFrame, filters: List[Dict[str, Any]], alias_map: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for f in filters:
        col = resolve_column(f.get("column"), alias_map, df)
        op = f.get("op")
        val = f.get("value")
        if not col or op not in SUPPORTED_OPERATORS:
            continue
        series = out[col]

        if op == "==":
            out = out[series == val]
        elif op == "!=":
            out = out[series != val]
        elif op in [">", ">=", "<", "<="]:
            s = _safe_numeric(series)
            try:
                v = float(val)
            except Exception:
                v = pd.NA
            if pd.isna(v):
                continue
            if op == ">":
                out = out[s > v]
            elif op == ">=":
                out = out[s >= v]
            elif op == "<":
                out = out[s < v]
            elif op == "<=":
                out = out[s <= v]
        elif op == "contains":
            out = out[series.astype(str).str.contains(str(val), case=False, na=False)]
        elif op == "in":
            values = set(_to_list_if_needed(val))
            out = out[series.isin(values)]
        elif op == "not in":
            values = set(_to_list_if_needed(val))
            out = out[~series.isin(values)]
    return out


def _status_mask_exited(status_series: pd.Series) -> pd.Series:
    s = status_series.astype(str).str.lower().fillna("")
    mask_any = pd.Series(False, index=s.index)
    for kw in EXIT_STATUS_KEYWORDS:
        mask_any = mask_any | s.str.contains(re.escape(kw.lower()), na=False)
    return mask_any


def _strip_status_filters(filters: List[Dict[str, Any]], alias_map: Dict[str, str], df: pd.DataFrame) -> List[Dict[str, Any]]:
    cleaned = []
    status_col = resolve_column("Employee Status", alias_map, df)
    for f in filters:
        col = resolve_column(f.get("column"), alias_map, df)
        if status_col and col == status_col:
            continue
        cleaned.append(f)
    return cleaned


def _strip_year_filters(filters: List[Dict[str, Any]], alias_map: Dict[str, str], df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Remove any Calendar Year-related filters from the list.
    Used when computing per-year metrics (e.g. turnover comparison).
    """
    cleaned = []
    year_col = resolve_column("Calendar Year", alias_map, df)
    for f in filters:
        col = resolve_column(f.get("column"), alias_map, df)
        if year_col and col == year_col:
            continue
        cleaned.append(f)
    return cleaned


def _extract_years_from_filters(filters: List[Dict[str, Any]], alias_map: Dict[str, str], df: pd.DataFrame) -> List[int]:
    """
    Extract all year values from filters where column is Calendar Year.
    Supports == and in.
    """
    years = set()
    year_col = resolve_column("Calendar Year", alias_map, df)
    if not year_col:
        return []

    for f in filters:
        col = resolve_column(f.get("column"), alias_map, df)
        if col != year_col:
            continue

        op = f.get("op")
        val = f.get("value")

        if op == "==":
            try:
                years.add(int(val))
            except Exception:
                continue
        elif op == "in":
            vals = _to_list_if_needed(val)
            for v in vals:
                try:
                    years.add(int(v))
                except Exception:
                    continue

    return sorted(years)


# Execution helpers

def _compute_headcount_unique(df: pd.DataFrame, id_col: str) -> int:
    return df[id_col].nunique()


def _compute_turnover_rate(
    df: pd.DataFrame,
    base_filters: List[Dict[str, Any]],
    alias_map: Dict[str, str],
    id_col: str,
) -> Tuple[float, int, int]:
    status_col = resolve_column("Employee Status", alias_map, df)
    if not status_col:
        raise ValueError("Could not find 'Employee Status' column to compute turnover.")
    denom_filters = _strip_status_filters(base_filters, alias_map, df)
    denom_df = apply_filters(df, denom_filters, alias_map)
    denom = _compute_headcount_unique(denom_df, id_col)
    if denom == 0:
        return 0.0, 0, 0
    exit_mask = _status_mask_exited(denom_df[status_col])
    numer = denom_df.loc[exit_mask, id_col].nunique()
    return numer / denom, numer, denom


def _compute_turnover_comparison(
    df: pd.DataFrame,
    base_filters: List[Dict[str, Any]],
    alias_map: Dict[str, str],
    id_col: str,
) -> str:
    """
    Compare turnover between two (or more) years and return a natural language sentence,
    e.g. 'Turnover increased by 3.2 percentage points (10.5% → 13.7%).'
    """
    years = _extract_years_from_filters(base_filters, alias_map, df)
    if len(years) < 2:
        return "To compare turnover, please specify at least two years."

    # Use the earliest and latest years for the comparison
    start_year = years[0]
    end_year = years[-1]

    year_col = resolve_column("Calendar Year", alias_map, df)
    if not year_col:
        return "Could not find 'Calendar Year' column to compare turnover."

    # Common filters without any year restriction
    base_without_year = _strip_year_filters(base_filters, alias_map, df)

    def turnover_for_year(y: int) -> Tuple[float, int, int]:
        year_filter = {"column": year_col, "op": "==", "value": y}
        filters_y = base_without_year + [year_filter]
        return _compute_turnover_rate(df, filters_y, alias_map, id_col)

    try:
        rate_start, num_start, denom_start = turnover_for_year(start_year)
        rate_end, num_end, denom_end = turnover_for_year(end_year)
    except Exception as e:
        return f"Could not compute turnover comparison: {e}"

    pct_start = rate_start * 100
    pct_end = rate_end * 100
    diff = pct_end - pct_start

    direction = "increased" if diff > 0 else "decreased" if diff < 0 else "stayed the same"
    diff_abs = abs(diff)

    if direction == "stayed the same":
        return (
            f"Between {start_year} and {end_year}, turnover stayed the same "
            f"at {pct_start:.2f}%."
        )

    return (
        f"Between {start_year} and {end_year}, turnover {direction} "
        f"by {diff_abs:.2f} percentage points "
        f"({pct_start:.2f}% → {pct_end:.2f}%)."
    )


def _compute_headcount_comparison(
    df: pd.DataFrame,
    base_filters: List[Dict[str, Any]],
    alias_map: Dict[str, str],
    id_col: str,
) -> str:
    """
    Compare headcount (unique employees) between two (or more) years.
    Example output:
    'Between 2023 and 2025, headcount increased by 120 (4500 → 4620).'
    """
    years = _extract_years_from_filters(base_filters, alias_map, df)
    if len(years) < 2:
        return "To compare headcount, please specify at least two years."

    # earliest & latest year
    start_year = years[0]
    end_year = years[-1]

    year_col = resolve_column("Calendar Year", alias_map, df)
    if not year_col:
        return "Could not find 'Calendar Year' column to compare headcount."

    # remove year filters
    base_without_year = _strip_year_filters(base_filters, alias_map, df)

    def headcount_for_year(y: int) -> int:
        year_filter = {"column": year_col, "op": "==", "value": y}
        filters_y = base_without_year + [year_filter]
        filtered = apply_filters(df, filters_y, alias_map)
        return _compute_headcount_unique(filtered, id_col)

    try:
        hc_start = headcount_for_year(start_year)
        hc_end = headcount_for_year(end_year)
    except Exception as e:
        return f"Could not compute headcount comparison: {e}"

    diff = hc_end - hc_start
    if diff > 0:
        direction = "increased"
    elif diff < 0:
        direction = "decreased"
    else:
        direction = "stayed the same"

    if direction == "stayed the same":
        return (
            f"Between {start_year} and {end_year}, headcount stayed the same "
            f"at {hc_start}."
        )

    return (
        f"Between {start_year} and {end_year}, headcount {direction} "
        f"by {abs(diff)} ({hc_start} → {hc_end})."
    )



def _format_markdown_table(df: pd.DataFrame, index_name: Optional[str] = None) -> str:
    if df.empty:
        return "_No rows._"
    if index_name and df.index.name != index_name:
        df = df.copy()
        df.index.name = index_name
    return df.to_markdown()


def describe_filters(filters: List[Dict[str, Any]], alias_map: Dict[str, str], df: pd.DataFrame) -> str:
    """
    Turn filters into a simple natural-language phrase.
    Example: 'for Calendar Year 2025 and Department Department 3'
    """
    if not filters:
        return "overall"

    pieces = []
    for f in filters:
        col_raw = f.get("column")
        col = resolve_column(col_raw, alias_map, df) or col_raw
        op = f.get("op")
        val = f.get("value")

        if op == "==" and col:
            pieces.append(f"{col} {val}")
        elif op == "in" and col:
            vals = _to_list_if_needed(val)
            vals_str = ", ".join(map(str, vals))
            pieces.append(f"{col} in [{vals_str}]")
        else:
            if col:
                pieces.append(f"{col} {op} {val}")
            else:
                pieces.append(f"{col_raw} {op} {val}")

    return "for " + " and ".join(pieces)


# Enhanced Gender Analysis Functions

def _compute_gender_share(
    df: pd.DataFrame,
    id_col: str,
    alias_map: Dict[str, str],
    show_percentage: bool = True,
    show_count: bool = False,
) -> str:
    """
    Compute overall gender distribution.
    """
    gender_col = resolve_column("Gender", alias_map, df)
    if not gender_col:
        return "Could not find 'Gender' column."

    if df.empty:
        return "No data available."

    gender_counts = df.groupby(gender_col, dropna=False).size()
    total = gender_counts.sum()

    if total == 0:
        return "No employees in the filtered data."

    gender_pct = (gender_counts / total * 100).round(2)

    result_data = {}
    if show_count:
        result_data["Count"] = gender_counts
    if show_percentage:
        result_data["Percentage"] = gender_pct.apply(lambda x: f"{x}%")

    if not show_count and not show_percentage:
        result_data["Percentage"] = gender_pct.apply(lambda x: f"{x}%")

    result = pd.DataFrame(result_data)
    result.index.name = "Gender"

    return _format_markdown_table(result, index_name="Gender")


def _compute_gender_comparison(
    df: pd.DataFrame,
    filters: List[Dict[str, Any]],
    alias_map: Dict[str, str],
    id_col: str,
    show_percentage: bool = True,
    show_count: bool = False,
) -> str:
    """
    Compare gender distribution across years.
    """
    gender_col = resolve_column("Gender", alias_map, df)
    year_col = resolve_column("Calendar Year", alias_map, df)

    if not gender_col:
        return "Could not find 'Gender' column."
    if not year_col:
        return "Could not find 'Calendar Year' column."

    non_year_filters = [
        f for f in filters
        if resolve_column(f.get("column"), alias_map, df) != year_col
    ]
    filtered_df = apply_filters(df, non_year_filters, alias_map)

    if filtered_df.empty:
        return "No data available after filtering."

    grouped = filtered_df.groupby([year_col, gender_col], dropna=False).size().unstack(fill_value=0)

    totals = grouped.sum(axis=1)
    pct_df = grouped.div(totals, axis=0) * 100
    pct_df = pct_df.round(2)

    final_cols = []
    gender_cols = list(grouped.columns)

    for g in gender_cols:
        if show_percentage:
            pct_df[f"{g}"] = pct_df[g].apply(lambda x: f"{x}%")
            final_cols.append(f"{g}")
        if show_count:
            pct_df[f"{g} Count"] = grouped[g]
            final_cols.append(f"{g} Count")

    if show_count:
        pct_df["Total"] = totals.astype(int)
        final_cols.append("Total")

    if not show_count and not show_percentage:
        for g in gender_cols:
            pct_df[f"{g}"] = (
                grouped[g] / totals * 100
            ).round(2).apply(lambda x: f"{x}%")
            final_cols.append(f"{g}")

    result = pct_df[final_cols]
    result.index.name = "Year"

    return _format_markdown_table(result, index_name="Year")


def _execute_gender_grouped(
    df: pd.DataFrame,
    filters: List[Dict[str, Any]],
    alias_map: Dict[str, str],
    id_col: str,
    group_by_cols: List[str],
    show_percentage: bool = True,
    show_count: bool = False,
) -> str:
    """
    Execute gender share analysis grouped by dimensions (e.g., Location, Department).
    """
    gender_col = resolve_column("Gender", alias_map, df)
    if not gender_col:
        return "Could not find 'Gender' column."

    group_cols_resolved = []
    for g in group_by_cols:
        resolved = resolve_column(g, alias_map, df)
        if resolved and resolved != gender_col:
            group_cols_resolved.append(resolved)

    if not group_cols_resolved:
        return "Could not resolve group_by columns (excluding Gender)."

    filtered_df = apply_filters(df, filters, alias_map)
    if filtered_df.empty:
        return "No data after applying filters."

    all_groups = group_cols_resolved + [gender_col]
    grouped = filtered_df.groupby(all_groups, dropna=False).size().unstack(fill_value=0)

    totals = grouped.sum(axis=1)
    pct_df = grouped.div(totals, axis=0) * 100
    pct_df = pct_df.round(2)

    result = pd.DataFrame(index=grouped.index)
    gender_cols = list(grouped.columns)

    for col in gender_cols:
        if show_percentage:
            result[f"{col}"] = pct_df[col].apply(lambda x: f"{x}%")
        if show_count:
            result[f"{col} Count"] = grouped[col].astype(int)

    if show_count:
        result["Total"] = totals.astype(int)

    if not show_count and not show_percentage:
        for col in gender_cols:
            result[f"{col}"] = pct_df[col].apply(lambda x: f"{x}%")

    if len(group_cols_resolved) == 1:
        index_name = group_cols_resolved[0]
    else:
        index_name = ", ".join(group_cols_resolved)

    return _format_markdown_table(result, index_name=index_name)


# Grouped execution

def _execute_grouped(
    df: pd.DataFrame,
    filters: List[Dict[str, Any]],
    alias_map: Dict[str, str],
    metrics: List[Dict[str, Any]],
    group_by_cols: List[str],
) -> str:
    group_cols_resolved: List[str] = []
    for g in group_by_cols:
        resolved = resolve_column(g, alias_map, df)
        if resolved:
            group_cols_resolved.append(resolved)
    if not group_cols_resolved:
        return "Could not resolve group_by columns."

    has_gender_metric = any(m.get("name") in ["gender_share", "gender_comparison"] for m in metrics)
    if has_gender_metric:
        first_metric = metrics[0]
        id_col = resolve_column(first_metric.get("on", ""), alias_map, df) or pick_id_column(df)
        if not id_col:
            return "Could not resolve ID column."

        show_percentage = first_metric.get("show_percentage", True)
        show_count = first_metric.get("show_count", False)

        table = _execute_gender_grouped(
            df, filters, alias_map, id_col, group_cols_resolved,
            show_percentage, show_count
        )
        group_label = ", ".join(group_cols_resolved)
        intro = f"Here is the gender split by {group_label}:"
        return f"{intro}\n\n{table}"

    # Non-gender grouped (headcount / turnover)
    first_metric = metrics[0]
    id_col = resolve_column(first_metric.get("on", ""), alias_map, df) or pick_id_column(df)
    if not id_col:
        return "Could not resolve ID column."

    denom_filters = _strip_status_filters(filters, alias_map, df)
    denom_df = apply_filters(df, denom_filters, alias_map)
    if denom_df.empty:
        return "No data after applying filters."

    grouped = denom_df.groupby(group_cols_resolved, dropna=False)
    denom_series = grouped[id_col].nunique().rename("Headcount")

    status_col = resolve_column("Employee Status", alias_map, df)
    if not status_col:
        return "Could not find 'Employee Status' column to compute turnover."
    exit_mask = _status_mask_exited(denom_df[status_col])
    numerator_series = (
        denom_df
        .loc[exit_mask]
        .groupby(group_cols_resolved, dropna=False)[id_col]
        .nunique()
        .rename("Exited")
    )

    out = pd.concat([denom_series, numerator_series], axis=1).fillna(0)
    out["Headcount"] = out["Headcount"].astype(int)
    out["Exited"] = out["Exited"].astype(int)

    wants_turnover = any(m.get("name") == "turnover_rate" for m in metrics)
    wants_hc = any(m.get("name") == "headcount_unique" for m in metrics)

    cols = []
    if wants_hc:
        cols.append("Headcount")
    if wants_turnover:
        out["Turnover %"] = (
            out["Exited"] / out["Headcount"].where(out["Headcount"] != 0, 1) * 100
        ).round(2)
        cols.append("Exited")
        cols.append("Turnover %")

    if "Turnover %" in out.columns:
        out = out.sort_values("Turnover %", ascending=False)
    else:
        out = out.sort_values("Headcount", ascending=False)

    out = out[cols] if cols else out

    if len(group_cols_resolved) == 1:
        index_name = group_cols_resolved[0]
    else:
        index_name = ", ".join(group_cols_resolved)

    table = _format_markdown_table(out, index_name=index_name)
    group_label = ", ".join(group_cols_resolved)
    intro = f"Here is the result by {group_label}:"
    return f"{intro}\n\n{table}"

def sanitize_plan(plan, df):
    # If the query uses group_by Business Unit and filters include Business Unit that doesn't exist → remove it
    bu_col = "Business Unit"
    valid_bus = set(df[bu_col].unique())

    new_filters = []
    for f in plan.get("filters", []):
        if f["column"] == bu_col and f["value"] not in valid_bus:
            continue
        new_filters.append(f)

    plan["filters"] = new_filters
    return plan

# Main execution

def execute_plan(df: pd.DataFrame, plan: Dict[str, Any]) -> str:
    intent = plan.get("intent")
    if intent != "metric":
        return "Only metric intent is supported."

    filters = plan.get("filters", [])
    metrics = plan.get("metrics", [])
    group_by_cols = plan.get("group_by", [])

    if not metrics:
        return "No metrics provided."

    alias_map = build_alias_map(df)

    if group_by_cols:
        try:
            table_md = _execute_grouped(df, filters, alias_map, metrics, group_by_cols)
            return table_md
        except Exception as e:
            return f"Error computing grouped result: {e}"

    # Ungrouped (scalar) path
    outputs = []
    where_text = describe_filters(filters, alias_map, df)

    for m in metrics:
        name = m.get("name")
        id_col = resolve_column(m.get("on"), alias_map, df) or pick_id_column(df)
        if not id_col:
            return "Could not resolve ID column."
        alias = m.get("alias", name)

        show_percentage = m.get("show_percentage", True)
        show_count = m.get("show_count", False)

        if name == "headcount_unique":
            filtered = apply_filters(df, filters, alias_map)
            hc = _compute_headcount_unique(filtered, id_col)
            outputs.append(f"The number of headcount {where_text} is {hc}.")
        elif name == "turnover_rate":
            try:
                rate, numer, denom = _compute_turnover_rate(df, filters, alias_map, id_col)
            except Exception as e:
                outputs.append(f"Could not compute turnover rate {where_text}: {e}")
                continue
            pct = f"{rate*100:.2f}%"
            outputs.append(
                f"The turnover rate {where_text} is {pct} "
                f"(Exited: {numer} / Headcount: {denom})."
            )
        elif name == "turnover_comparison":
            result = _compute_turnover_comparison(df, filters, alias_map, id_col)
            outputs.append(result)
        elif name == "headcount_comparison":
            result = _compute_headcount_comparison(df, filters, alias_map, id_col)
            outputs.append(result)
        elif name == "gender_share":
            filtered = apply_filters(df, filters, alias_map)
            table = _compute_gender_share(filtered, id_col, alias_map, show_percentage, show_count)
            outputs.append(f"The gender split {where_text} is:\n\n{table}")
        elif name == "gender_comparison":
            table = _compute_gender_comparison(df, filters, alias_map, id_col, show_percentage, show_count)
            outputs.append(f"Here is the gender distribution by year:\n\n{table}")
        else:
            outputs.append(f"{alias}: unsupported metric '{name}'")

    return "\n\n".join(outputs)


def plan_with_ollama(question: str, df: pd.DataFrame) -> Dict[str, Any]:
    normalized_question = normalize_question(question)
    messages = make_planner_messages(normalized_question, df)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = ollama.chat(model=MODEL_NAME, messages=messages)
            content = resp.get("message", {}).get("content", "")
            return parse_plan(content)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                import time
                time.sleep(2)
            else:
                raise Exception(
                    f"Failed after {max_retries} attempts. Error: {e}\n\n"
                    "Please check:\n"
                    "1. Ollama server is running (run 'ollama serve')\n"
                    "2. Model is installed (run 'ollama pull {MODEL_NAME}')\n"
                    "3. System has enough memory"
                )


# Gradio UI

def load_dataframe(file: Union[str, None]) -> pd.DataFrame:
    if file.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def on_file_uploaded(file) -> Tuple[str, Dict[str, Any]]:
    df = load_dataframe(file.name)
    return f"Loaded {df.shape}", {"df": df}


def chat_fn(history, message, state):
    if history is None:
        history = []

    df = state.get("df")

    # Add user message to history
    history = history + [{"role": "user", "content": message}]

    if df is None:
        history = history + [
            {"role": "assistant", "content": "Please upload a file first."}
        ]
        return history, state

    try:
        plan = plan_with_ollama(message, df)
        answer = execute_plan(df, plan)
        bot = answer
    except Exception as e:
        bot = f"Error: {e}"

    history = history + [{"role": "assistant", "content": bot}]
    return history, state


with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown("""
    ### Supported Queries:
    - **Headcount**
      - "What is the headcount in 2025?"
      - "What is number of HC in 2025 in Department 3?"
      - "What is total number of headcount in Business Unit BU12 in 2025?"
      - "What is total headcount at location US in Department 1 in 2025?"
      - "Compare headcount in 2023 and 2025"
      - "Compare headcount between 2024 and 2025 in Location US"
      - "Compare headcount between 2024 and 2025 in Business Unit BU11"
    - **Turnover**
      - "What is the turnover rate in 2024?"
      - "Show turnover rate by Department in 2024"
      - "How did turnover rate change between 2024 and 2025?"
      - "Compare turnover in 2023 and 2025 in Department 3"
      - "Show turnover rate by Business Unit and Location in 2025"
    - **Gender Split (Overall)**
      - "What is the gender split in 2024?"
      - "What is gender diversity in 2025?"
    - **Gender Split (Grouped)**
      - "What is the gender split in 2024 by Location?"
      - "What is the gender diversity in 2024 by Business Unit?"
      - "Show gender split in 2024 by Department including counts"
      - "Show gender diversity in 2025 by Location with both percentage and count"
    - **Gender Comparison**
      - "Compare gender distribution between 2023 and 2025"
      - "Compare gender diversity between 2023 and 2025 in Location Mexico"
      - "Compare gender split between 2023 and 2025 for Business Unit BU11"
    """)
    file_u = gr.File(label="Upload HR CSV/XLSX", file_types=[".csv", ".xlsx"])
    status = gr.Markdown("No file uploaded yet.")
    chatbot = gr.Chatbot(height=360, type="messages")
    user_input = gr.Textbox(label="Ask a question")
    state = gr.State({"df": None})

    file_u.upload(fn=on_file_uploaded, inputs=[file_u], outputs=[status, state])
    user_input.submit(fn=chat_fn, inputs=[chatbot, user_input, state], outputs=[chatbot, state])

if __name__ == "__main__":
    demo.launch()