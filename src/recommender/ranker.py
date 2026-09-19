def rank_recommendations(gaps, top_k=5):
    """
    Ranks missing skills by their predicted score delta impact.
    
    Args:
        gaps (list of dict): The output from analyze_gaps().
        top_k (int): Number of top skills to return.
        
    Returns:
        list of dict: Top K skills sorted by delta descending.
    """
    if not gaps:
        return []
        
    # Sort descending by delta
    ranked = sorted(gaps, key=lambda x: x['delta'], reverse=True)
    
    return ranked[:top_k]
