#!/usr/bin/env python3
"""
Test game over detection.

Instructions:
1. Open Chrome Dino game (chrome://dino or disconnect internet)
2. Run this script with --select-region
3. The script will take random actions and monitor for game over
4. When dino crashes, it should detect game over and restart
"""

import time

from main import DinoEnv


def test_game_over_detection() -> None:
    """Test the game over detection mechanism."""
    print("=" * 60)
    print("GAME OVER DETECTION TEST")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Make sure Chrome Dino game is visible and ready")
    print("2. The test will take random actions until game over")
    print("3. It will detect game over and attempt to restart")
    print("=" * 60)

    # Create environment with region selection
    print("\nInitializing environment...")
    env = DinoEnv(select_region=True, game_over_threshold=0.99)

    # Run test episodes
    num_episodes = 3
    for episode in range(num_episodes):
        print(f"\n{'=' * 60}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print(f"{'=' * 60}")

        _obs, _info = env.reset()
        print("Game started/restarted. Taking random actions...")

        step_count = 0
        max_steps_per_episode = 500  # Safety limit

        while step_count < max_steps_per_episode:
            # Take random action
            action = env.action_space.sample()

            # Execute action
            _obs, reward, done, _truncated, _info = env.step(action)

            step_count += 1

            # Print status every 50 steps
            if step_count % 50 == 0:
                print(f"  Step {step_count}: Still alive...")

            # Check if game over detected
            if done:
                print(f"\n✓ Game over detected after {step_count} steps!")
                print(f"  Reward: {reward}")
                print("  Waiting 2 seconds before restart...")
                time.sleep(2)
                break

        if step_count >= max_steps_per_episode:
            print(f"\n! Reached max steps ({max_steps_per_episode}) without game over")
            print("  Either you're very lucky or threshold needs adjustment!")

    print(f"\n{'=' * 60}")
    print("TEST COMPLETE")
    print(f"{'=' * 60}")
    print("\nResults:")
    print(f"  Episodes completed: {num_episodes}")
    print("\nIf game over was detected each time, the implementation works!")


def test_no_false_positives() -> None:
    """Test that game over is not detected during normal gameplay."""
    print("\n" + "=" * 60)
    print("FALSE POSITIVE TEST")
    print("=" * 60)
    print("\nThis test checks that game over is NOT detected during normal play.")
    print("Make sure the game is running and the dino is NOT crashing!")
    print("=" * 60)

    env = DinoEnv(select_region=True, game_over_threshold=0.99)

    _obs, _info = env.reset()
    print("\nTaking 100 steps with no action (just monitoring)...")

    false_positives = 0
    for step in range(100):
        # Take no action (action=0)
        _obs, _reward, done, _truncated, _info = env.step(0)

        if done:
            false_positives += 1
            print(f"  ! False positive at step {step}")
            # Reset to continue test
            _obs, _info = env.reset()

        if step % 20 == 0:
            print(f"  Step {step}/100...")

    print(f"\n{'=' * 60}")
    print("FALSE POSITIVE TEST COMPLETE")
    print(f"{'=' * 60}")
    print(f"False positives detected: {false_positives}")

    if false_positives == 0:
        print("✓ No false positives! Game over detection is working correctly.")
    else:
        print("✗ False positives detected. Consider lowering threshold or increasing num_frames_to_compare.")


def manual_test() -> None:
    """Interactive test where you can manually verify game over detection."""
    print("\n" + "=" * 60)
    print("MANUAL GAME OVER TEST")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Start the game and let it run")
    print("2. When you're ready, let the dino crash")
    print("3. Watch the console to see when game over is detected")
    print("=" * 60)

    env = DinoEnv(select_region=True, game_over_threshold=0.99)

    _obs, _info = env.reset()
    print("\nMonitoring for game over... (Press Ctrl+C to stop)")

    try:
        step = 0
        while True:
            # Take no action, just monitor
            _obs, _reward, done, _truncated, _info = env.step(0)

            step += 1

            if done:
                print(f"\n{'=' * 60}")
                print(f"✓ GAME OVER DETECTED at step {step}!")
                print(f"{'=' * 60}")
                print("Restarting in 2 seconds...")
                time.sleep(2)
                _obs, _info = env.reset()
                step = 0
                print("Game restarted. Monitoring again...")
            elif step % 30 == 0:
                print(f"Step {step}: Game still running...")

            time.sleep(0.1)  # Slow down for visibility

    except KeyboardInterrupt:
        print("\n\nTest stopped by user.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test game over detection")
    parser.add_argument(
        "--test",
        choices=["all", "detection", "false-positive", "manual"],
        default="all",
        help="Which test to run",
    )
    args = parser.parse_args()

    if args.test == "all":
        test_game_over_detection()
        input("\nPress ENTER to continue to false positive test...")
        test_no_false_positives()
    elif args.test == "detection":
        test_game_over_detection()
    elif args.test == "false-positive":
        test_no_false_positives()
    elif args.test == "manual":
        manual_test()
