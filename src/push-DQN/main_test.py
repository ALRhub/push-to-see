from src.mask_rg.mask_rg import MaskRG
import time
import os
import random
import threading
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import cv2
from collections import namedtuple
import torch
from torch.autograd import Variable
from robot import Robot
from trainer import Trainer
from logger import Logger
import utils
import scipy.misc
import yaml


CONF_PATH = './push-DQN_config.yaml'

class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def main():

    with open(CONF_PATH) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # plt.ion()
    mask_rg = MaskRG(config['detection_thresholds']['confidence_threshold'], config['detection_thresholds']['mask_threshold'])
    session_success_threshold = config['detection_thresholds']['session_success_threshold']

    snapshot_file = os.path.join(config['model']['path'], config['model']['file']) if config['model']['file'] != 'new' else None
    #TODO clean this from trainer method
    load_snapshot = True if snapshot_file is not None else False

    logging_directory = config['logging']['path'] if config['logging']['path'] != 'Default' else os.path.abspath('./logs')
    continue_logging = config['logging']['continue_logging']
    save_visualizations = config['logging']['save_visualizations']

    workspace_limits = np.asarray(config['environment']['workspace_limits'])
    heightmap_resolution = float(config['environment']['heightmap_resolution'])

    min_num_obj = config['environment']['min_num_objects']
    max_num_obj = config['environment']['max_num_objects']
    session_limit = config['environment']['session_limit']

    random_seed = 1234 # (to set the random seed for simulation and neural net initialization)
    force_cpu = False # (to force code to run in CPU mode)

    # TODO implementation!
    # experience_replay = config['setup']['experience_replay']

    future_reward_discount = config['training']['future_reward_discount']

    is_testing = config['setup']['is_testing']
    explore_prob = config['setup']['exploration_probability']
    explore_rate_decay = config['setup']['epsilon_greedy_policy']['exploration_rate_decay']
    min_rate_decay = config['setup']['epsilon_greedy_policy']['min_exp_rate_decay']

    is_random_model = config['setup']['is_random_model']

    # Set random seed
    np.random.seed(random_seed)

    # Initialize pick-and-place system (camera and robot)
    robot = Robot(min_num_obj, max_num_obj, workspace_limits)

    # Initialize trainer
    trainer = Trainer(future_reward_discount, is_testing, load_snapshot, snapshot_file, force_cpu)

    # Initialize data logger
    logger = Logger(continue_logging, logging_directory)
    logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose, robot.cam_depth_scale) # Save camera intrinsics and pose
    logger.save_heightmap_info(workspace_limits, heightmap_resolution) # Save heightmap parameters

    # save the current configuration file in the current logging directory
    logger.save_config_file(CONF_PATH)

    # Find last executed iteration of pre-loaded log, and load execution info and RL variables
    if continue_logging:
        trainer.preload(logger.transitions_directory)

    # Initialize variables for heuristic bootstrapping and exploration probability
    # no_change_count = [2, 2] if not is_testing else [0, 0]

    explore_prob = explore_prob if not is_testing else 0.0
    # This overrides all above!!!
    if is_random_model:
        explore_prob = 1.0
        explore_rate_decay = False

    nonlocal_variables = {'executing_action': False,
                          'primitive_action': None,
                          'best_pix_ind': None,
                          'push_success': False,
                          'grasp_success': False,
                          'session_success': False,
                          'session_first_loop': True}
    session_counter = 0
    # session no (index)--> [number of iteration for that session, fail(0)/success(1)]
    success_fail_record = []
    last_iter = 0
    # plt.ion()

    # Parallel thread to process network output and execute actions
    # -------------------------------------------------------------
    def process_actions():
        while True:
            if nonlocal_variables['executing_action']:

                print('Primitive confidence scores for pushing: {:f}'.format(np.max(push_predictions)))
                nonlocal_variables['primitive_action'] = 'push'

                # Exploration vs. Exploitation - Best push vs random push / if is_testing --> explore_prob=0!
                explore_actions = np.random.uniform() < explore_prob
                if not explore_actions:
                    print('Strategy: ' + BColors.WARNING + 'EXPLOIT' + BColors.ENDC + ' (exploration probability: '
                          + BColors.OKBLUE + '{:f})'.format(explore_prob) + BColors.ENDC)
                    # Exploitation: Return the index of the x, y coordinates of the best prediction in the Q-values
                    nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
                    predicted_value = np.max(push_predictions)

                else:
                    print('Strategy:' + BColors.WARNING + 'EXPLORE' + BColors.ENDC + ' (exploration probability:'
                          + BColors.OKBLUE + '{:f})'.format(explore_prob) + BColors.ENDC)

                    nonlocal_variables['best_pix_ind'] = (np.random.random_integers(0, 15),
                                                          np.random.random_integers(0, 224-1),
                                                          np.random.random_integers(0, 224-1))
                    # nonlocal_variables['best_pix_ind']  = np.random.random_integers(0, 224-1)
                    # nonlocal_variables['best_pix_ind'] = np.random.random_integers(0, 224-1)
                    predicted_value = push_predictions[nonlocal_variables['best_pix_ind']]

                trainer.is_exploit_log.append([0 if explore_actions else 1])
                logger.write_to_log('is-exploit', trainer.is_exploit_log)
                # Save predicted confidence value
                trainer.predicted_value_log.append([predicted_value])
                logger.write_to_log('predicted-value', trainer.predicted_value_log)

                # Compute 3D position of pixel
                print('Action: %s at (%d, %d, %d)' % (nonlocal_variables['primitive_action'], nonlocal_variables['best_pix_ind'][0],
                                                      nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]))

                best_rotation_angle = np.deg2rad(nonlocal_variables['best_pix_ind'][0] * (360.0/trainer.model.num_rotations))
                best_pix_x = nonlocal_variables['best_pix_ind'][2]
                best_pix_y = nonlocal_variables['best_pix_ind'][1]
                primitive_position = [best_pix_x * heightmap_resolution + workspace_limits[0][0],
                                      best_pix_y * heightmap_resolution + workspace_limits[1][0],
                                      valid_depth_heightmap[best_pix_y][best_pix_x] + workspace_limits[2][0]]
                print("primitive position --> ", primitive_position)
                # If pushing, adjust start position, and make sure z value is safe and not too low
                if nonlocal_variables['primitive_action'] == 'push':
                    finger_width = 0.02
                    safe_kernel_width = int(np.round((finger_width/2)/heightmap_resolution))
                    local_region = valid_depth_heightmap[max(best_pix_y - safe_kernel_width, 0)
                                                         :min(best_pix_y + safe_kernel_width + 1, valid_depth_heightmap.shape[0]),
                                                                max(best_pix_x - safe_kernel_width, 0)
                                                                :min(best_pix_x + safe_kernel_width + 1, valid_depth_heightmap.shape[1])]
                    if local_region.size == 0:
                        safe_z_position = workspace_limits[2][0]

                    elif (np.max(local_region) - workspace_limits[2][0]) > finger_width:
                        safe_z_position = np.max(local_region) - finger_width/4
                    else:
                        safe_z_position = np.max(local_region)

                    primitive_position[2] = safe_z_position

                # Save executed primitive --> 0 for exploit, 1 for explore
                if not explore_actions:
                    trainer.executed_action_log.append([0, nonlocal_variables['best_pix_ind'][0],
                                                        nonlocal_variables['best_pix_ind'][1],
                                                        nonlocal_variables['best_pix_ind'][2]])
                else:
                    trainer.executed_action_log.append([1, nonlocal_variables['best_pix_ind'][0],
                                                        nonlocal_variables['best_pix_ind'][1],
                                                        nonlocal_variables['best_pix_ind'][2]])

                logger.write_to_log('executed-action', trainer.executed_action_log)

                # Visualize executed primitive, and affordances
                if save_visualizations:
                    push_pred_vis = trainer.get_prediction_vis(push_predictions, color_heightmap, nonlocal_variables['best_pix_ind'])
                    logger.save_visualizations(trainer.iteration, push_pred_vis, 'push')
                    cv2.imwrite('visualization.push.png', push_pred_vis)

                # Initialize variables that influence reward
                nonlocal_variables['push_success'] = False

                # robot.move_to_target([-0.5, 0.0, 0.036 + safe_z_position])

                # Execute primitive
                if nonlocal_variables['primitive_action'] == 'push':
                    nonlocal_variables['push_success'] = robot.push(primitive_position, best_rotation_angle, workspace_limits)
                    print('Push executed: %r' % (nonlocal_variables['push_success']))
                print(BColors.OKBLUE + '-------------------------------------------------------------------------'
                      + BColors.ENDC)
                nonlocal_variables['executing_action'] = False


            time.sleep(0.01)
    action_thread = threading.Thread(target=process_actions)
    action_thread.daemon = True
    action_thread.start()
    exit_called = False
    # -------------------------------------------------------------
    # -------------------------------------------------------------

    # Start main training/testing loop
    while True:
        print(BColors.WARNING + '\n%s iteration: %d' % ('Testing' if is_testing else 'Training', trainer.iteration) +
              BColors.ENDC + '  (Session iteration: {:d})'.format(session_counter))
        iteration_time_0 = time.time()
        session_counter += 1
        # Make sure simulation is still stable (if not, reset simulation)
        robot.check_sim()

        # Get latest RGB-D image
        color_img, depth_img_raw = robot.get_camera_data()
        # Detph scale is 1 for simulation!!
        depth_img = depth_img_raw * robot.cam_depth_scale # Apply depth scale from calibration

        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics,
                                                               robot.cam_pose, workspace_limits, heightmap_resolution)
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

        # CHECK IF IT IS THE FIRST LOOP
        if nonlocal_variables['session_first_loop']:

            print(BColors.WARNING + 'This is the zeroth iteration! '
                                    'Mask-RG Values below will be used in the first iteration!' + BColors.ENDC)
            # Get ground truth segmentation masks
            color_m_rg, depth_m_rg, [segmentation_mask, num_objects] = robot.get_data_mask_rg()
            # print(num_objects)
            plt.imsave('ground_truth.png', segmentation_mask)
            plt.imsave('color_img.png', color_m_rg)
            plt.imsave('depth_img.png', depth_m_rg)

            # set the mask rcnn reward generator with the current depth and gt
            mask_rg.set_reward_generator(depth_m_rg, segmentation_mask)

            # get the rewards predictions
            pred_ids, seg_reward, err_rate = mask_rg.get_current_rewards()
            printout = mask_rg.print_segmentation(pred_ids)
            plt.imsave('mask_pred_diff.png', printout)

            prev_seg_reward = seg_reward.copy()
            prev_depth_for_chg_det = depth_m_rg.copy()
            depth_m_rg_act = depth_m_rg.copy() # for change detection patch

            # if there is a success without a push just by chance, flag for the termination of the session
            if seg_reward > session_success_threshold:
                success_without_push = True
                nonlocal_variables['session_success'] = True
            else:
                success_without_push = False

            # END OF FIRST LOOP --> DEACTIVATE
            nonlocal_variables['session_first_loop'] = False

        # Save RGB-D images and RGB-D heightmaps
        logger.save_images(trainer.iteration, color_img, depth_img, '0')
        logger.save_heightmaps(trainer.iteration, color_heightmap, valid_depth_heightmap, '0')

        # Save mask prediction difference image and error rate per iteration
        if 'color_m_rg_act' in locals():
            logger.save_mask_diff_image(printout, segmentation_mask_action, color_m_rg_act,
                                        depth_m_rg_act, trainer.iteration, session_counter-1, err_rate)

        if not exit_called:
            # Run forward pass with network to get affordances
            push_predictions = trainer.forward(color_heightmap, valid_depth_heightmap, is_volatile=True)

            # Execute best primitive action on robot in another thread
            nonlocal_variables['executing_action'] = True

        # Run training iteration in current thread (aka training thread)
        # at the first loop prev_color_img will not be in locals! as well as other vars with 'prev_'
        if 'prev_color_img' in locals():
            # The thresh hold that will terminate the current session with success, this line is after current
            # push action and getting the current scene reward score from mask_rg
            if seg_reward > session_success_threshold:
                # Push success!!!
                nonlocal_variables['session_success'] = True

            # Detect changes
            depth_diff = abs(depth_heightmap - prev_depth_heightmap)
            # depth_diff = abs(depth_m_rg_act - prev_depth_for_chg_det)
            depth_diff[np.isnan(depth_diff)] = 0
            depth_diff[depth_diff < 0.01] = 0

            depth_diff[depth_diff > 0] = 1
            change_threshold = 100
            change_value = np.sum(depth_diff)
            change_detected = change_value > change_threshold
            print('Change detected: %r (value: %d)' % (change_detected, change_value))

            if not success_without_push:
                label_value, prev_reward_value, seg_vals = trainer.get_reward_value(color_heightmap, valid_depth_heightmap,
                                                                      prev_seg_reward, seg_reward, change_detected)

                trainer.label_value_log.append([label_value])
                logger.write_to_log('label-value', trainer.label_value_log)
                trainer.reward_value_log.append([prev_reward_value])
                logger.write_to_log('reward-value', trainer.reward_value_log)

                logger.save_mask_rg_returns([seg_vals, num_objects_action.size-1])

        # Sync both action thread and training thread
        while nonlocal_variables['executing_action']:
            time.sleep(0.01)

        # SUCCESS
        if nonlocal_variables['session_success']:
            if not success_without_push:
                success_fail_record.append([session_counter, 1])
                logger.save_session_success_fail(' Session no ' + str(len(success_fail_record)) + ' --> after '
                                                 + str(session_counter - 1) +
                                                 ' pushing actions --> SUCCESS!  (tr_it --> {:d})'.format(trainer.iteration))

                print(BColors.WARNING + 'SESSION SUCCESS!' + BColors.ENDC)
            else:
                success_fail_record.append([session_counter, 2])
                logger.save_session_success_fail(' Session no ' + str(len(success_fail_record)) + ' --> after '
                                                 + str(session_counter - 1) +
                                                 ' TRAINING IGNORED --> (tr_it --> {:d})'.format(
                                                     trainer.iteration))
                print(BColors.WARNING + 'SESSION CLOSED WITHOUT TRAINING AS THE INITIAL SEGMENTATION WAS GOOD ENOUGH!'
                      + BColors.ENDC)

            last_iter = trainer.iteration
            nonlocal_variables['session_success'] = False
            nonlocal_variables['session_first_loop'] = True
            seg_reward = np.asarray(0)
            session_counter = 0
            robot.restart_sim()

        # FAIL
        # it depends how to count --> 0th iteration of a session does not do any training, just gets two GTs to be able
        #  to calculate rewards at the 1st iteration. "session_counter > 30" means after 31 sessions including the 0th
        #  session or 30 training sessions
        if session_counter > session_limit:

            success_fail_record.append([session_counter, 0])
            logger.save_session_success_fail('Session no ' + str(len(success_fail_record)) + ' --> after '
                                             + str(session_counter) +
                                             ' pushing actions --> FAIL!  (tr_it --> {:d})'.format(trainer.iteration))
            last_iter = trainer.iteration
            nonlocal_variables['session_first_loop'] = True
            robot.restart_sim()
            # robot.add_objects(18)
            seg_reward = np.asarray(0)

            print(BColors.WARNING + 'SESSION TIMEOUT! (after 30 iterations without success)' + BColors.ENDC)
            session_counter = 0

        if exit_called:
            break

        # Save information for next training step
        prev_color_img = color_img.copy()
        prev_depth_img = depth_img.copy()
        prev_color_heightmap = color_heightmap.copy()
        prev_depth_heightmap = depth_heightmap.copy()
        prev_valid_depth_heightmap = valid_depth_heightmap.copy()
        prev_push_success = nonlocal_variables['push_success']
        prev_grasp_success = False
        prev_primitive_action = 'push'
        prev_push_predictions = push_predictions.copy()
        prev_best_pix_ind = nonlocal_variables['best_pix_ind']


        if not nonlocal_variables['session_first_loop']:
            # Action is done
            prev_seg_reward = seg_reward.copy()
            prev_depth_for_chg_det = depth_m_rg_act.copy()
            # print('Previous segmentation reward --> {:.5f}'.format(prev_seg_reward))
            # Get RGB-D image
            color_img_action, depth_img_raw_action = robot.get_camera_data()
            # Get ground truth segmentation masks
            color_m_rg_act, depth_m_rg_act, [segmentation_mask_action, num_objects_action] = robot.get_data_mask_rg()
            plt.imsave('ground_truth.png', segmentation_mask_action)
            plt.imsave('color_img.png', color_m_rg_act)
            plt.imsave('depth_img.png', depth_m_rg_act)
            mask_rg.set_reward_generator(depth_m_rg_act, segmentation_mask_action)

            pred_ids, seg_reward, err_rate = mask_rg.get_current_rewards()
            printout = mask_rg.print_segmentation(pred_ids)
            plt.imsave('mask_pred_diff.png', printout)
            print('Number of the objects left in the scene after pushing action -->', num_objects_action.size - 1)
            # plot_obj.set_data(printout)
            # plt.draw()
            all_masks = mask_rg.print_masks()
            plt.imsave('mask_pred_all.png', all_masks)

        trainer.iteration += 1
        iteration_time_1 = time.time()
        print('Time elapsed: %f' % (iteration_time_1-iteration_time_0))
        print('---------------------------------------------------------------')
        print('---------------------------------------------------------------')


if __name__ == '__main__':
    main()
