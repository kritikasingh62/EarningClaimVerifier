"""
Company-agnostic metric mappings for claim extraction and verification.

Extend these mappings to improve accuracy across any company.
"""

# Evidence patterns -> preferred SEC metrics (extractor reclassification)
# Add patterns when LLM misclassifies claims. Format: (pattern_keywords, [metric_preference_list])
COST_REVENUE_PATTERNS = [
    "cost of revenue",
    "cost of revenues",
    "cor ",
    "total cost",
    "other cost of revenue",
    "cost of goods",
]
CAPEX_PATTERNS = [
    "capex",
    "cap ex",
    "capital expenditure",
    "capital spending",
    "investments in production",
    "facilities and data center",
]

# Metric equivalence for verification fallback (when primary gives huge delta)
# Primary metric -> alternate metrics that measure same/similar concept
METRIC_EQUIVALENTS = {
    "us-gaap:SalesRevenueNet": ["us-gaap:Revenues", "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"],
    "us-gaap:Revenues": ["us-gaap:SalesRevenueNet", "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"],
    "us-gaap:SalesRevenueServicesNet": ["us-gaap:Revenues", "us-gaap:SalesRevenueNet"],
    "us-gaap:CostOfRevenue": ["us-gaap:CostOfGoodsAndServicesSold"],
    "us-gaap:CostOfGoodsAndServicesSold": ["us-gaap:CostOfRevenue"],
}

# Revenue metrics to try for margin computation (first available wins)
REVENUE_METRICS_FOR_MARGIN = [
    "us-gaap:Revenues",
    "us-gaap:SalesRevenueNet",
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
]

# CapEx metrics (PaymentsToAcquire...)
CAPEX_METRICS = [
    "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
    "us-gaap:PaymentsToAcquirePropertyPlantAndEquipmentAndOtherProductiveAssets",
]
