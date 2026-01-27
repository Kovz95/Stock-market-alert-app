"""Test that backend.py functions can be imported successfully."""

try:
    from backend import (
        evaluate_expression_list,
        simplify_conditions,
        indicator_calculation,
        evaluate_expression,
        apply_function,
        ind_to_dict
    )
    
    print("[OK] All functions imported successfully!")
    print("\nAvailable functions:")
    print("  - evaluate_expression_list")
    print("  - simplify_conditions")
    print("  - indicator_calculation")
    print("  - evaluate_expression")
    print("  - apply_function")
    print("  - ind_to_dict")
    
    # Test simplify_conditions
    print("\n--- Testing simplify_conditions ---")
    result = simplify_conditions("Close[-1] > 150")
    print(f"simplify_conditions('Close[-1] > 150'): {result}")
    
    # Test ind_to_dict
    print("\n--- Testing ind_to_dict ---")
    result = ind_to_dict("Close[-1]")
    print(f"ind_to_dict('Close[-1]'): {result}")
    
    result = ind_to_dict("rsi(14)[-1]")
    print(f"ind_to_dict('rsi(14)[-1]'): {result}")
    
    result = ind_to_dict("150")
    print(f"ind_to_dict('150'): {result}")
    
    print("\n[OK] All tests passed!")
    
except Exception as e:
    print(f"[ERROR] Error importing or testing: {e}")
    import traceback
    traceback.print_exc()
