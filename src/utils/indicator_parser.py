import copy

def extract_indicators(parsed_condition):
    """
    Extracts unique indicators and their parameters from the parsed condition.

    Args:
        parsed_condition (list): The parsed condition tree.

    Returns:
        set: A set of unique indicators with their parameters.
    """
    work_copy = copy.deepcopy(parsed_condition)
    indicators = set()

    def recursive_extract(node):
        print("DEBUG node:", node)
        # Base case: Check if the node represents an indicator
        if isinstance(node, list) and len(node) > 0 and isinstance(node[0], str):
            indicator_name = node[0]
            if indicator_name in {
                "rsi", "bb", "sma", "hma", "ema", "slope_sma", "slope_ema", "slope_hma", "stochastic", "macd", "vwap",
                "atr", "obv", "cmf", "psar", "cci", "williamsr", "roc"
            }:
                for i in node:
                    if '[' in i:
                        node.remove(i)
                        continue
                indicators.add(tuple(node))  # Add the full indicator info as a tuple
        # Recursive case: Traverse deeper into the tree
            elif indicator_name == 'breakout':
                for child in node:
                    recursive_extract(child)
        elif isinstance(node, list):
            for child in node:
                recursive_extract(child)

    # Start recursive traversal
    recursive_extract(work_copy)
    print("debug "+str(indicators))
    return indicators
