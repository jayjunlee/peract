import logging
from typing import List

import numpy as np
from rlbench.demo import Demo


def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped

def is_gripper_open(gripper_joint_positions):
    if gripper_joint_positions[0] < 0.0399 and gripper_joint_positions[1] < 0.0399:
        return False
    else:
        return True

def keypoint_discovery(demo: Demo,
                       stopping_delta=0.1,
                       method='heuristic') -> List[int]:
    episode_keypoints = []
    if method == 'heuristic':
        prev_gripper_open = demo[0].gripper_open
        stopped_buffer = 0
        for i, obs in enumerate(demo):
            print(i, obs.gripper_open)
            print(dir(obs))

            stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
            stopped_buffer = 4 if stopped else stopped_buffer - 1
            # If change in gripper, or end of episode.
            last = i == (len(demo) - 1)
            if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
                episode_keypoints.append(i)
            prev_gripper_open = obs.gripper_open
        if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == episode_keypoints[-2]:
            episode_keypoints.pop(-2)
        logging.debug(f'Found {len(episode_keypoints)} keypoints.')
        return episode_keypoints

    elif method == 'random':
        # Randomly select keypoints.
        episode_keypoints = np.random.choice(
            range(len(demo)),
            size=20,
            replace=False)
        episode_keypoints.sort()
        return episode_keypoints

    elif method == 'fixed_interval':
        # Fixed interval.
        episode_keypoints = []
        segment_length = len(demo) // 20
        for i in range(0, len(demo), segment_length):
            episode_keypoints.append(i)
        return episode_keypoints

    else:
        raise NotImplementedError

def is_gripper_open_v2(gripper_touch_forces, gripper_open):
    if np.sum(np.abs(gripper_touch_forces)) > 0.7 and gripper_open==1.0:
        return False
    else:
        return True

def keypoint_discovery_v2(demo: Demo,
                       stopping_delta=0.1,
                       method='heuristic') -> List[int]:
    episode_keypoints = []
    if method == 'heuristic':
        prev_gripper_open = demo[0].gripper_open
        stopped_buffer = 0
        force_vec = np.array([0, 0, 0, 0 ,0 , 0])
        for i, obs in enumerate(demo):
            stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
            stopped_buffer = 4 if stopped else stopped_buffer - 1
            # If change in gripper, or end of episode.
            last = i == (len(demo) - 1)
            # print(dir(obs))
            print(i, np.sum(np.abs(obs.gripper_touch_forces)), obs.gripper_open)
            gripper_open = is_gripper_open(obs.gripper_joint_positions)
            if np.sum(np.abs(obs.gripper_touch_forces)) > 0.7 and gripper_open is True:
                demo[i].gripper_open = False
                gripper_open = False
            if i != 0 and (gripper_open != prev_gripper_open or last or stopped):
                # print(obs.gripper_open, prev_gripper_open)
                episode_keypoints.append(i)
            prev_gripper_open = gripper_open
        if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == episode_keypoints[-2]:
            episode_keypoints.pop(-2)
        logging.debug(f'Found {len(episode_keypoints)} keypoints.')
        return episode_keypoints
    else:
        raise NotImplementedError

# find minimum difference between any two elements in list
def find_minimum_difference(lst):
    minimum = lst[-1]
    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] < minimum:
            minimum = lst[i] - lst[i - 1]
    return minimum