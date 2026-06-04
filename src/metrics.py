"""Compute derived financial metrics from extracted filing figures."""


def operating_margin(operating_income: float, revenue: float) -> float:
    return operating_income / revenue


def net_margin(net_income: float, revenue: float) -> float:
    return net_income / revenue


def revenue_growth(current_revenue: float, prior_revenue: float) -> float:
    return (current_revenue - prior_revenue) / prior_revenue


def free_cash_flow(operating_cash_flow: float, capital_expenditures: float) -> float:
    return operating_cash_flow - abs(capital_expenditures)

