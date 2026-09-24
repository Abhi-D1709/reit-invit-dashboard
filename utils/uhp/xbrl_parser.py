"""Generic parser for SEBI Unit Holding Pattern (REIT/InvIT) XBRL instance documents."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

NS = {
    "xbrli": "http://www.xbrl.org/2003/instance",
    "xbrldi": "http://xbrl.org/2006/xbrldi",
}
CAPMKT_NS = "https://www.sebi.gov.in/xbrl/2022-03-31/in-capmkt"


@dataclass
class ParsedUHP:
    contexts: dict = field(default_factory=dict)
    facts: dict = field(default_factory=dict)

    def get(self, context_id: str, field_name: str, default=None):
        return self.facts.get(context_id, {}).get(field_name, default)

    def has(self, context_id: str) -> bool:
        return context_id in self.facts


MEMBER_ALIASES = {"ClearingMembers": "Clearings"}


def _canonical_context_id(raw_id: str, member: str | None) -> str:
    """NSE and BSE name their contexts differently but use the same category
    members, so contexts are keyed by member: '<Member>I' (NSE's convention).
    """
    if member:
        name = member.split(":")[-1].removesuffix("Member")
        return MEMBER_ALIASES.get(name, name) + "I"
    if raw_id == "MainI":
        return "OneI"
    return raw_id


def parse_uhp_xbrl(xml_text: str | bytes) -> ParsedUHP:
    root = ET.fromstring(xml_text)

    contexts: dict[str, dict] = {}
    canonical: dict[str, str] = {}
    for c in root.findall("xbrli:context", NS):
        raw_id = c.get("id")
        dim_el = c.find(".//xbrldi:explicitMember", NS)
        instant_el = c.find(".//xbrli:instant", NS)
        member = dim_el.text if dim_el is not None else None
        cid = _canonical_context_id(raw_id, member)
        canonical[raw_id] = cid
        contexts[cid] = {
            "member": member,
            "instant": instant_el.text if instant_el is not None else None,
        }

    facts: dict[str, dict[str, str]] = {}
    prefix = "{" + CAPMKT_NS + "}"
    for el in root:
        tag = el.tag
        if not tag.startswith(prefix):
            continue
        local = tag[len(prefix):]
        cref = el.get("contextRef")
        if not cref:
            continue
        facts.setdefault(canonical.get(cref, cref), {})[local] = el.text

    _normalize_percent_scale(facts)
    return ParsedUHP(contexts=contexts, facts=facts)


def _normalize_percent_scale(facts: dict[str, dict[str, str]]) -> None:
    """NSE files report Table I percentages as 0-100 (total = 100); BSE files
    report fractions (total = 1). Rescale fractions so both read as percent."""
    total = facts.get("TotalUnitsOutstandingI", {}).get("AsAPercentageOfTotalOutStandingUnits")
    if total is None or not 0 < to_number(total) <= 1.5:
        return
    for fields in facts.values():
        for key, value in fields.items():
            if key.startswith("AsAPercentage") and value not in (None, ""):
                fields[key] = str(round(to_number(value) * 100, 10))


def to_number(value, default=0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
