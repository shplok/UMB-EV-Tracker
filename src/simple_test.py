"""
Simple Test Case - Import, Initialize, and Print Parameters

This is the most basic test to verify EVTracker installation.
"""

from ev_tracker import EVTracker

print("="*70)
print("SIMPLE TEST: Import, Initialize, and Print Parameters")
print("="*70)

# Step 1: Import (already done above)
print("\n✓ Step 1: Import successful")
print("  from ev_tracker import EVTracker")

# Step 2: Initialize tracker
tracker = EVTracker()
print("\n✓ Step 2: Tracker initialized")
print("  tracker = EVTracker()")

# Step 3: Print default parameters
print("\n✓ Step 3: Default parameters:")
tracker.print_params()

# Step 4: Change some parameters
print("\n✓ Step 4: Setting custom parameters...")
tracker.set_params(
    threshold=0.6,
    min_distance=35,
    max_distance=30
)
print("  tracker.set_params(threshold=0.6, min_distance=35, max_distance=30)")

# Step 5: Print updated parameters
print("\n✓ Step 5: Updated parameters:")
tracker.print_params()

print("\n" + "="*70)
print("✅ SUCCESS! EVTracker is working correctly.")
print("="*70)
print("\nNext steps:")
print("  1. Run full test suite: python test_all_features.py")
print("  2. Analyze your data: tracker.run('movie.tif', 'ground_truth.csv')")
print("="*70)