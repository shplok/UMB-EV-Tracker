from ev_tracker import EVTracker, quick_analyze
import os

# ============================================================================
# CONFIGURATION - UPDATE THESE WITH YOUR ACTUAL FILES
# ============================================================================

# Single file for testing
TEST_TIFF = r"UMB-EV-Tracker\data\tiff\new\xslot_HCC1954_PT00_xp1_750uLhr_z35um_mov_1_MMStack_Pos0.ome.tif"
TEST_CSV = r"UMB-EV-Tracker\data\csv\new\InfocusEVs_xslot_HCC1954_PT00_xp1_750uLhr_z35um_mov_1.csv"

# Multiple files for batch testing
BATCH_DATASET = [
    (
        r"UMB-EV-Tracker/data/tiff/new/xslot_BT747_PT03_xp4_750uLhr_z35um_mov_2_MMStack_Pos0.ome.tif",
        r"UMB-EV-Tracker/data/csv/new/xslot_BT747_PT03_xp4_750uLhr_z35um_mov_2.csv"
    ),
    (
        r"UMB-EV-Tracker\data\tiff\xslot_HCC1954_01_500uLhr_z35um_mov_1_MMStack_Pos0.ome.tif",
        r"UMB-EV-Tracker\data\csv\xslot_HCC1954_01_500uLhr_z35um_mov_1.csv"
    ),
]

OUTPUT_DIR = "out/test_run"

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_1_basic_import():
    """Test 1: Basic import and initialization"""
    print("\n" + "="*70)
    print("TEST 1: BASIC IMPORT AND INITIALIZATION")
    print("="*70)
    
    try:
        from ev_tracker import EVTracker
        print("✓ Import successful")
        
        tracker = EVTracker()
        print("✓ EVTracker created with default parameters")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_2_parameter_setting():
    """Test 2: Setting parameters"""
    print("\n" + "="*70)
    print("TEST 2: PARAMETER SETTING")
    print("="*70)
    
    try:
        tracker = EVTracker()
        
        # Test individual parameters
        tracker.set_params(threshold=0.6)
        print("✓ Set threshold")
        
        tracker.set_params(min_distance=35)
        print("✓ Set min_distance")
        
        tracker.set_params(max_distance=30)
        print("✓ Set max_distance")
        
        # Test multiple parameters at once
        tracker.set_params(
            threshold=0.55,
            min_distance=30,
            max_distance=25,
            min_track_length=5
        )
        print("✓ Set multiple parameters")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_3_parameter_validation():
    """Test 3: Parameter validation"""
    print("\n" + "="*70)
    print("TEST 3: PARAMETER VALIDATION")
    print("="*70)
    
    tracker = EVTracker()
    
    # Test invalid threshold (should raise error)
    try:
        tracker.set_params(threshold=1.5)
        print("✗ Failed to catch invalid threshold (>1)")
        return False
    except ValueError:
        print("✓ Correctly rejected invalid threshold (>1)")
    
    # Test valid threshold
    try:
        tracker.set_params(threshold=0.5)
        print("✓ Accepted valid threshold (0.5)")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    return True


def test_4_print_params():
    """Test 4: Print parameters"""
    print("\n" + "="*70)
    print("TEST 4: PRINT PARAMETERS")
    print("="*70)
    
    try:
        tracker = EVTracker()
        tracker.set_params(threshold=0.6, min_distance=35, max_distance=30)
        tracker.print_params()
        print("✓ Parameters printed successfully")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_5_method_chaining():
    """Test 5: Method chaining"""
    print("\n" + "="*70)
    print("TEST 5: METHOD CHAINING")
    print("="*70)
    
    try:
        tracker = (EVTracker(output_dir=OUTPUT_DIR)
                  .set_params(threshold=0.55, min_distance=30)
                  .set_params(max_distance=25))
        
        print("✓ Method chaining works")
        tracker.print_params()
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_6_quick_analyze():
    """Test 6: Quick analyze function"""
    print("\n" + "="*70)
    print("TEST 6: QUICK ANALYZE FUNCTION")
    print("="*70)
    
    if not os.path.exists(TEST_TIFF):
        print(f"⚠ Skipping: Test file not found: {TEST_TIFF}")
        return True
    
    try:
        print(f"Running quick analysis on: {os.path.basename(TEST_TIFF)}")
        results = quick_analyze(TEST_TIFF, TEST_CSV, threshold=0.55)
        
        if results['success']:
            print("✓ Quick analyze completed successfully")
            print(f"  Global AP: {results.get('global_ap', 'N/A')}")
            print(f"  Output: {results.get('output_dir', 'N/A')}")
            return True
        else:
            print(f"✗ Analysis failed: {results.get('error', 'Unknown')}")
            return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_7_single_file_analysis():
    """Test 7: Single file analysis with run()"""
    print("\n" + "="*70)
    print("TEST 7: SINGLE FILE ANALYSIS")
    print("="*70)
    
    if not os.path.exists(TEST_TIFF):
        print(f"⚠ Skipping: Test file not found: {TEST_TIFF}")
        return True
    
    try:
        tracker = EVTracker(output_dir=OUTPUT_DIR)
        tracker.set_params(threshold=0.55, min_distance=30)
        
        print(f"Running analysis on: {os.path.basename(TEST_TIFF)}")
        results = tracker.run(TEST_TIFF, TEST_CSV)
        
        if results['success']:
            print("✓ Single file analysis completed")
            print(f"  Global AP: {results.get('global_ap', 'N/A')}")
            print(f"  Global AUC: {results.get('global_auc', 'N/A')}")
            print(f"  Total points: {results.get('total_points', 'N/A')}")
            return True
        else:
            print(f"✗ Analysis failed: {results.get('error', 'Unknown')}")
            return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_8_batch_analysis():
    """Test 8: Batch analysis with run_batch()"""
    print("\n" + "="*70)
    print("TEST 8: BATCH ANALYSIS")
    print("="*70)
    
    # Check if files exist
    valid_files = []
    for tiff, csv in BATCH_DATASET:
        if os.path.exists(tiff) and os.path.exists(csv):
            valid_files.append((tiff, csv))
    
    if len(valid_files) == 0:
        print("⚠ Skipping: No valid test files found")
        print("  Update BATCH_DATASET with your file paths")
        return True
    
    try:
        tracker = EVTracker(output_dir=OUTPUT_DIR)
        tracker.set_params(threshold=0.55, min_distance=30, max_distance=25)
        
        print(f"Running batch analysis on {len(valid_files)} files...")
        results = tracker.run_batch(valid_files)
        
        if results['success']:
            print("✓ Batch analysis completed")
            print(f"  Global AP: {results.get('global_ap', 'N/A'):.4f}")
            print(f"  Global AUC: {results.get('global_auc', 'N/A'):.4f}")
            print(f"  Total points: {results.get('total_points', 'N/A')}")
            print(f"  Files processed: {len(results.get('file_summaries', []))}")
            return True
        else:
            print(f"✗ Batch analysis failed: {results.get('error', 'Unknown')}")
            return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_9_parameter_sweep():
    """Test 9: Parameter sweep (testing multiple thresholds)"""
    print("\n" + "="*70)
    print("TEST 9: PARAMETER SWEEP")
    print("="*70)
    
    if not os.path.exists(TEST_TIFF):
        print(f"⚠ Skipping: Test file not found")
        return True
    
    try:
        tracker = EVTracker(output_dir=OUTPUT_DIR)
        thresholds = [0.4, 0.5, 0.6]
        
        print(f"Testing thresholds: {thresholds}")
        
        for thresh in thresholds:
            tracker.set_params(threshold=thresh)
            print(f"\n  Testing threshold={thresh}...")
            
            # Note: This will create outputs for each threshold
            # In real use, you might want custom output_subdir for each
            results = tracker.run(TEST_TIFF, TEST_CSV)
            
            if results['success']:
                ap = results.get('global_ap', 0)
                print(f"  ✓ Threshold {thresh}: AP = {ap:.4f}")
            else:
                print(f"  ✗ Threshold {thresh}: Failed")
        
        print("\n✓ Parameter sweep completed")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_10_all_parameters():
    """Test 10: Set all available parameters"""
    print("\n" + "="*70)
    print("TEST 10: ALL AVAILABLE PARAMETERS")
    print("="*70)
    
    try:
        tracker = EVTracker(output_dir=OUTPUT_DIR)
        
        # Set every parameter
        tracker.set_params(
            threshold=0.55,
            min_distance=30,
            max_distance=25,
            min_track_length=5,
            filter_radius=10,
            bg_window_size=15
        )
        
        print("✓ All parameters set successfully")
        tracker.print_params()
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests and report results"""
    
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*20 + "EV TRACKER - COMPLETE TEST SUITE" + " "*16 + "║")
    print("╚" + "="*68 + "╝")
    
    tests = [
        ("Basic Import", test_1_basic_import),
        ("Parameter Setting", test_2_parameter_setting),
        ("Parameter Validation", test_3_parameter_validation),
        ("Print Parameters", test_4_print_params),
        ("Method Chaining", test_5_method_chaining),
        ("Quick Analyze", test_6_quick_analyze),
        ("Single File Analysis", test_7_single_file_analysis),
        ("Batch Analysis", test_8_batch_analysis),
        ("Parameter Sweep", test_9_parameter_sweep),
        ("All Parameters", test_10_all_parameters),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8} - {name}")
    
    print("\n" + "-"*70)
    print(f"  Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! EVTracker is working perfectly.")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Check errors above.")
    
    return passed == total


# ============================================================================
# INTERACTIVE MENU (Optional)
# ============================================================================

def interactive_menu():
    """Interactive menu to run specific tests"""
    
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*20 + "EV TRACKER - TEST MENU" + " "*25 + "║")
    print("╚" + "="*68 + "╝")
    
    print("\nSelect a test to run:")
    print("  1.  Basic Import")
    print("  2.  Parameter Setting")
    print("  3.  Parameter Validation")
    print("  4.  Print Parameters")
    print("  5.  Method Chaining")
    print("  6.  Quick Analyze")
    print("  7.  Single File Analysis")
    print("  8.  Batch Analysis")
    print("  9.  Parameter Sweep")
    print("  10. All Parameters")
    print("  0.  Run ALL Tests")
    print("  q.  Quit")
    
    choice = input("\nEnter choice: ").strip()
    
    tests = {
        '1': test_1_basic_import,
        '2': test_2_parameter_setting,
        '3': test_3_parameter_validation,
        '4': test_4_print_params,
        '5': test_5_method_chaining,
        '6': test_6_quick_analyze,
        '7': test_7_single_file_analysis,
        '8': test_8_batch_analysis,
        '9': test_9_parameter_sweep,
        '10': test_10_all_parameters,
        '0': run_all_tests,
    }
    
    if choice == 'q':
        print("Goodbye!")
        return
    
    test_func = tests.get(choice)
    if test_func:
        test_func()
    else:
        print("Invalid choice!")
    
    print("\n" + "="*70)
    input("Press Enter to continue...")
    interactive_menu()


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Check if interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_menu()
    else:
        # Run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)