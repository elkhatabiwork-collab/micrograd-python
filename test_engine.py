from micrograd.engine import Value

def test_value_operators():
    print("--- Running Micrograd Engine Test ---")

    # 1. Initialize custom Value objects
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    
    # 2. Forward Pass: Compute g = (a * b) + c
    raw_sum = a * b + c          # Expected: (2.0 * -3.0) + 10.0 = 4.0
    g = raw_sum.tanh()           # Expected: tanh(4.0) ≈ 0.999329
    
    print(f"Forward Pass Result (g): {g.data:.6f}")
    
    # 3. Backward Pass: Compute all derivatives
    g.backward()
    
    # 4. Print results to verify gradients
    print("\nCalculated Gradients:")
    print(f"dg/dg (Should be 1.0): {g.grad}")
    print(f"dg/dc (Should be 1 - tanh^2(4) ≈ 0.00134): {c.grad:.6f}")
    print(f"dg/da (Should be b * dg/dc ≈ -0.00402):  {a.grad:.6f}")
    print(f"dg/db (Should be a * dg/dc ≈ 0.00268):   {b.grad:.6f}")

if __name__ == "__main__":
    test_value_operators()