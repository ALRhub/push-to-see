import time
import numpy as np
import utils
from simulation import vrep
import matplotlib.pyplot as plt

class Robot(object):
    def __init__(self, min_num_obj, max_num_obj, workspace_limits):

        self.workspace_limits = workspace_limits

        self.min_num_obj = min_num_obj  # 18 # 18/24 or 14/20
        self.max_num_obj = max_num_obj  # 24

        # dropping params
        self.drop_limits = np.asarray([[-0.6, -0.4], [-0.15, 0.15], [-0.2, -0.1]])
        self.DROP_HEIGHT = 0.2

        # Make sure to have the server side running in V-REP:
        # in a child script of a V-REP scene, add following command
        # to be executed just once, at simulation start:
        #
        # simExtRemoteApiStart(19999)
        #
        # then start simulation, and run this program.
        #
        # IMPORTANT: for each successful call to simxStart, there
        # should be a corresponding call to simxFinish at the end!

        # MODIFY remoteApiConnections.txt

        # Connect to simulator
        vrep.simxFinish(-1) # Just in case, close all opened connections
        self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5) # Connect to V-REP on port 19997
        if self.sim_client == -1:
            print('Failed to connect to simulation (V-REP remote API server). Exiting.')
            exit()
        else:
            print('Connected to simulation.')
            self.restart_sim()

        # Setup virtual camera in simulation
        self.setup_sim_camera()

    def setup_sim_camera(self):

        # Get handle to camera
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp', vrep.simx_opmode_blocking)

        sim_ret_cam_ortho, self.cam_handle_ortho = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_ortho', vrep.simx_opmode_blocking)

        # Get handles for masking
        sim_ret_cam_gt, self.cam_handle_gt = vrep.simxGetObjectHandle(self.sim_client, 'gt_sensor',
                                                                            vrep.simx_opmode_blocking)

        # Get camera pose and intrinsics in simulation
        sim_ret, cam_position = vrep.simxGetObjectPosition(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        cam_trans = np.eye(4, 4)
        cam_trans[0:3, 3] = np.asarray(cam_position)
        # WHY minus?
        cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        cam_rotm = np.eye(4, 4)
        cam_rotm[0:3, 0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
        self.cam_pose = np.dot(cam_trans, cam_rotm) # Compute rigid transformation representating camera pose
        self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.cam_depth_scale = 1

        # Get background image
        self.bg_color_img, self.bg_depth_img = self.get_camera_data()
        self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale

    def restart_sim(self):

        sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(1) # this gives enough time to Vrep to reset object dynamics in the simulation so that mask_rg can get the depth at the zeroth iteration!
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(1)
        sim_ret, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        while gripper_position[2] > 0.4: # V-REP bug requiring multiple starts and stops to restart
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)

        # this is necessary for the fixed test scenes (or not)
        # self.setup_sim_camera()
        # time.sleep(1)

        # set all objects back to static and unrespondable
        list_of_shapes = ['convex_{:d}'.format(i) for i in range(0, 72)]
        # list_of_shapes = ['imported_part_{:d}'.format(i) for i in range(0, 26)]

        # ret = vrep.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer', vrep.sim_scripttype_childscript,
        #                                   'setStatic', [0], [0], list_of_shapes, bytearray(), vrep.simx_opmode_blocking)

        # add objects
        time.sleep(0.05)
        np.random.seed()
        curr_num_obj = np.random.random_integers(self.min_num_obj, self.max_num_obj)
        print('NUMBER OF OBJECTS DROPPED (range: {:d} - {:d}) --> {:d},'.format(self.min_num_obj, self.max_num_obj,
                                                                               curr_num_obj))
        # self.add_objects(curr_num_obj)
        asdasd= 1



    def check_sim(self):

        # Check if simulation is stable by checking if gripper is within workspace
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        sim_ok = gripper_position[0] > self.workspace_limits[0][0] - 0.1 and gripper_position[0] < self.workspace_limits[0][1] + 0.1 \
                 and gripper_position[1] > self.workspace_limits[1][0] - 0.1 and gripper_position[1] < self.workspace_limits[1][1] + 0.1 \
                 and gripper_position[2] > self.workspace_limits[2][0] and gripper_position[2] < self.workspace_limits[2][1]
        if not sim_ok:
            print('Simulation unstable. Restarting environment.')
            self.restart_sim()
            self.setup_sim_camera()

    def get_obj_positions(self):

        obj_positions = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)

        return obj_positions

    def get_obj_positions_and_orientations(self):

        obj_positions = []
        obj_orientations = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            sim_ret, object_orientation = vrep.simxGetObjectOrientation(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)
            obj_orientations.append(object_orientation)

        return obj_positions, obj_orientations

    def _get_segmentation_mask(self, mask_cam_handle):

        ret, res, data = vrep.simxGetVisionSensorImage(self.sim_client, mask_cam_handle, 0,
                                                      vrep.simx_opmode_oneshot_wait)

        seg_mask_temp = np.reshape(data, (res[1], res[0], 3))
        seg_mask_temp = seg_mask_temp[:, :, :1]
        seg_mask_temp = np.reshape(seg_mask_temp, (res[1], res[0]))
        seg_mask = seg_mask_temp.copy()

        # set the background pixels to 0
        elements, counts = np.unique(seg_mask, return_counts=True)
        background_index = np.argmax(counts)
        new_mask = np.where(seg_mask == elements[background_index], 0, seg_mask)

        objects = np.delete(elements, background_index)
        objects = np.sort(objects)

        for i in range(0, objects.size):
            # if not objects[i] == 0:
            new_mask[np.where(seg_mask == objects[i])] = i + 1

        new_mask = np.flip(new_mask, axis=0)
        objs = np.unique(new_mask)
        return new_mask, objs

    # def _get_segmentation_mask(self, mask_cam_handle):
    #
    #     # if self.plane_handle == 0 or self.floor_handle == 0:
    #
    #     # Get handle ids of the floor and the plane (at the moment Plane --> 19, Floor --> 10).
    #     # The related pixels will be removed whilst calculating the ground segmentation
    #     # ret_plane, self.plane_handle = vrep.simxGetObjectHandle(self.sim_client, 'Plane', vrep.simx_opmode_blocking)
    #     #
    #     # ret_floor, self.floor_handle = vrep.simxGetObjectHandle(self.sim_client, 'ResizableFloor_5_25_visibleElement',
    #     #                                              vrep.simx_opmode_blocking)
    #
    #
    #         # if not (ret_plane == 0 or ret_floor == 0):
    #     # self.plane_handle = 19
    #     # self.floor_handle = 10
    #
    #     ret, res, data = vrep.simxGetVisionSensorImage(self.sim_client, mask_cam_handle, 0,
    #                                                   vrep.simx_opmode_oneshot_wait)
    #
    #     time.sleep(0.01)
    #     # print('PING TIME -->  ', vrep.simxGetPingTime(self.sim_client))
    #
    #     seg_mask_temp = np.reshape(data, (res[1], res[0], 3))
    #     seg_mask_temp = seg_mask_temp[:, :, :1]
    #     seg_mask_temp = np.reshape(seg_mask_temp, (res[1], res[0]))
    #     asd = np.unique(seg_mask_temp)
    #     seg_mask = seg_mask_temp.copy()
    #     # print('Object handle ids  -->  ', asd, ' TOTAL = ', asd.size)
    #
    #     seg_mask = self._remove_bg_gt(seg_mask)
    #     objects = np.unique(seg_mask)
    #     objects = np.sort(objects)
    #     for i in range(0, objects.size):
    #         if not objects[i] == 0:
    #             seg_mask[np.where(seg_mask == objects[i])] = i
    #
    #     seg_mask = np.flip(seg_mask, axis=1)
    #     # includes 0 as background
    #     objs = np.unique(seg_mask)
    #     return seg_mask, objs

    # @staticmethod
    # def _remove_bg_gt(segmask):
    #     # ret_plane, plane_handle = vrep.simxGetObjectHandle(self.sim_client, 'Plane', vrep.simx_opmode_blocking)
    #     #
    #     # ret_floor, floor_handle = vrep.simxGetObjectHandle(self.sim_client, 'ResizableFloor_5_25_visibleElement',
    #     #                                              vrep.simx_opmode_blocking)
    #
    #     # remove two elements that cover max numbers of pixels (plane and floor)
    #     # just the background!
    #     elements, counts = np.unique(segmask, return_counts=True)
    #     background_index = np.argmax(counts)
    #     new_mask = np.where(segmask == elements[background_index], 0, segmask)
    #
    #     # this is not necessary as plane is removed from sensor
    #     # counts[background_index] = 0
    #     # background_index = np.argmax(counts)
    #     # new_mask = np.where(new_mask == elements[background_index], 0, new_mask)
    #     return new_mask

    def add_objects(self, num_obj_heap):

        # 18 INRIA objects x 4 = 72 (convex_0 to convex_71)
        isOscillating = 0
        objects = np.random.choice(range(0, 72), num_obj_heap, replace=False)
        # 6*10 --> basic meshes
        # objects = np.random.choice(range(0, 60), num_obj_heap, replace=False)
        # 2*13 --> adversarial
        # objects = np.random.choice(range(0, 26), num_obj_heap, replace=False)
        shapes = []
        objects_info = []

        for object_idx in objects:

            drop_x = (self.drop_limits[0][1] - self.drop_limits[0][0]) * np.random.random_sample() + \
                      self.drop_limits[0][0]
            drop_y = (self.drop_limits[1][1] - self.drop_limits[1][0]) * np.random.random_sample() + \
                     self.drop_limits[1][0]

            object_position = [drop_x, drop_y, self.DROP_HEIGHT]
            object_orientation = [2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample(),
                                  2 * np.pi * np.random.random_sample()]

            # str format for inria models
            shape_name = 'convex_{:d}'.format(object_idx)
            # str format for basic shapes
            # shape_name = 'imported_part_{:d}'.format(object_idx)

            shapes.append(shape_name)
            objects_info.append([shape_name, object_position, object_orientation])

            ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',
                                                                                                  vrep.sim_scripttype_childscript,
                                                                                                  'moveShape',
                                                                                                  [0],
                                                                                                  object_position + object_orientation,
                                                                                                  [shape_name],
                                                                                                  bytearray(),
                                                                                                  vrep.simx_opmode_blocking)
            # break if error
            if ret_resp == 8:
                print('Failed to add new objects to simulation! (Check shape names are correct!)')
                time.sleep(0.5)
                return [-1, -1]

            # wait a tad before dropping the next object
            time.sleep(0.04)

        # wait 100ms after all objects were released
        time.sleep(0.1)

        # wait until the oscillating objects stopped
        # (the remote method (i.e. checkMotion) that is attached to the vrep script (simulation_random_datacollect.ttt)
        # checks the angular velocities of the dropped objects)
        while isOscillating:
            ret_r, ret_i, ret_f, ret_s, ret_b = vrep.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',
                                                                            vrep.sim_scripttype_childscript,
                                                                            'checkMotion',
                                                                            [len(shapes)],
                                                                            [0.0],
                                                                            shapes,
                                                                            bytearray(),
                                                                            vrep.simx_opmode_blocking)
            time.sleep(0.05)
            isOscillating = ret_i[0]

        return [0, objects_info]

    def get_data_mask_rg(self):

        # Get color image from simulation --> for mask_rg
        sim_ret, resolution, raw_image_gt = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle_ortho, 0,
                                                                          vrep.simx_opmode_blocking)
        color_img_m_rg = np.asarray(raw_image_gt)
        color_img_m_rg.shape = (resolution[1], resolution[0], 3)
        color_img_m_rg = color_img_m_rg.astype(np.float) / 255
        color_img_m_rg[color_img_m_rg < 0] += 1
        color_img_m_rg *= 255
        # color_img_m_rg = np.fliplr(color_img_m_rg)
        color_img_m_rg = np.flip(color_img_m_rg, axis=0)
        color_img_m_rg = color_img_m_rg.astype(np.uint8)

        # Get depth image from simulation
        sim_ret, resolution, depth_buffer_gt = vrep.simxGetVisionSensorDepthBuffer(self.sim_client,
                                                                                   self.cam_handle_ortho,
                                                                                   vrep.simx_opmode_blocking)
        depth_img_m_rg = np.asarray(depth_buffer_gt)
        depth_img_m_rg.shape = (resolution[1], resolution[0])
        # depth_img_m_rg = np.fliplr(depth_img_m_rg)
        depth_img_m_rg = np.flip(depth_img_m_rg, axis=0)
        zNear_gt = 0.01  # 0.01
        zFar_gt = 0.5  # 10
        depth_img_m_rg = depth_img_m_rg * (zFar_gt - zNear_gt) + zNear_gt

        depth_img_m_rg = np.round(depth_img_m_rg * 10000).astype(np.uint16)

        # Get ground truth segmentation masks
        seg_mask = self._get_segmentation_mask(self.cam_handle_gt)

        return color_img_m_rg, depth_img_m_rg, seg_mask

    def get_camera_data(self):

        # Get color image from simulation
        sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0, vrep.simx_opmode_blocking)
        color_img = np.asarray(raw_image)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float)/255
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = np.fliplr(color_img)
        color_img = color_img.astype(np.uint8)

        # Get depth image from simulation
        sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle, vrep.simx_opmode_blocking)
        depth_img = np.asarray(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.fliplr(depth_img)
        zNear = 0.01
        zFar = 10
        depth_img = depth_img * (zFar - zNear) + zNear

        return color_img, depth_img


    def close_gripper(self, asynch=False):

        gripper_motor_velocity = -0.5
        gripper_motor_force = 100
        sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
        sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
        gripper_fully_closed = False
        while gripper_joint_position > -0.045: # Block until gripper is fully closed
            sim_ret, new_gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
            # print(gripper_joint_position)
            if new_gripper_joint_position >= gripper_joint_position:
                return gripper_fully_closed
            gripper_joint_position = new_gripper_joint_position
        gripper_fully_closed = True
        return gripper_fully_closed


    def open_gripper(self, asynch=False):
        gripper_motor_velocity = 0.5
        gripper_motor_force = 20
        sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
        sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
        while gripper_joint_position < 0.03: # Block until gripper is fully open
            sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)


    def move_to(self, tool_position, tool_orientation):

        # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)

        move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.02*move_direction/move_magnitude
        num_move_steps = int(np.floor(move_magnitude/0.02))

        for step_iter in range(num_move_steps):
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0], UR5_target_position[1] + move_step[1], UR5_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client,self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)


    def move_to_target(self, position):
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, position,
                                   vrep.simx_opmode_blocking)

    def push(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: push at (%f, %f, %f)' % (position[0], position[1], position[2]))

        # Compute tool orientation from heightmap rotation angle
        tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2

        # Adjust pushing point to be on tip of finger
        # there is another adjustment on z-axis below, too many adjustments... (pushing_point_margin). fadAlso in the main file (local_region)
        position[2] = position[2] + 0.026 # the adjustment value in the original code is +0.026

        # Compute pushing direction
        push_orientation = [1.0, 0.0]
        push_direction = np.asarray([push_orientation[0]*np.cos(heightmap_rotation_angle) - push_orientation[1]*np.sin(heightmap_rotation_angle),
                                     push_orientation[0]*np.sin(heightmap_rotation_angle) + push_orientation[1]*np.cos(heightmap_rotation_angle)])
        print("Pushing direction --> ", push_direction)

        # Move gripper to location above pushing point
        pushing_point_margin = 0.1
        location_above_pushing_point = (position[0], position[1], position[2] + pushing_point_margin)

        # Compute gripper position and linear movement increments #todo
        tool_position = location_above_pushing_point
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)

        if sim_ret != -1:
        # TODO check this part - in the first iterationValueError: cannot convert float NaN to integer --> previous sess_it 29 Success!
        # if move_step[0] is too close to zero then Python might turn it into zero because of decimal approximation!

            move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.05*move_direction/move_magnitude
            num_move_steps = int(np.floor(move_direction[0]/move_step[0]))

            # Compute gripper orientation and rotation increments
            sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/rotation_step))

            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                           (UR5_target_position[0] + move_step[0] * min(step_iter, num_move_steps),
                                            UR5_target_position[1] + move_step[1] * min(step_iter, num_move_steps),
                                            UR5_target_position[2] + move_step[2] * min(step_iter, num_move_steps)),
                                           vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                              (np.pi/2, gripper_orientation[1] + rotation_step * min(step_iter, num_rotation_steps),
                                               np.pi/2), vrep.simx_opmode_blocking)

            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                       (tool_position[0], tool_position[1], tool_position[2]), vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                          (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

            # Ensure gripper is closed
            self.close_gripper()

            # Approach pushing point
            self.move_to(position, None)

            # Compute target location (push to the right)
            push_length = 0.1
            target_x = min(max(position[0] + push_direction[0] * push_length, workspace_limits[0][0]), workspace_limits[0][1])
            target_y = min(max(position[1] + push_direction[1] * push_length, workspace_limits[1][0]), workspace_limits[1][1])
            #TODO check
            push_length = np.sqrt(np.power(target_x-position[0], 2) + np.power(target_y-position[1], 2))

            # Move in pushing direction towards target location
            self.move_to([target_x, target_y, position[2]], None)

            # Move gripper to location above grasp target
            self.move_to([target_x, target_y, location_above_pushing_point[2]], None)

            push_success = True
        else:
            push_success = False

        return push_success




























# JUNK

# command = "movel(p[%f,%f,%f,%f,%f,%f],0.5,0.2,0,0,a=1.2,v=0.25)\n" % (-0.5,-0.2,0.1,2.0171,2.4084,0)

# import socket

# HOST = "192.168.1.100"
# PORT = 30002
# s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# s.connect((HOST,PORT))

# j0 = 0
# j1 = -3.1415/2
# j2 = 3.1415/2
# j3 = -3.1415/2
# j4 = -3.1415/2
# j5 = 0;

# joint_acc = 1.2
# joint_vel = 0.25

# # command = "movej([%f,%f,%f,%f,%f,%f],a=%f,v=%f)\n" % (j0,j1,j2,j3,j4,j5,joint_acc,joint_vel)



# #


# # True closes
# command = "set_digital_out(8,True)\n"

# s.send(str.encode(command))
# data = s.recv(1024)



# s.close()
# print("Received",repr(data))





# print()

# String.Format ("movej([%f,%f,%f,%f,%f, %f], a={6}, v={7})\n", j0, j1, j2, j3, j4, j5, a, v);