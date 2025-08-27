import asyncio
import cv2
import apriltag
import numpy as np
import logging


class ApriltagWorker:
    def __init__(self, num_workers=2):
        self.num_workers = num_workers
        self.semaphore = asyncio.Semaphore(num_workers)
        self.apriltag_options = apriltag.DetectorOptions(families='tag36h11')
        self.apriltag_detector = apriltag.Detector(self.apriltag_options)

    async def detect_apriltag(self, frame, camera_params=None, tag_size=None, z_sign=1, camera_position=None):
        async with self.semaphore:
            return await asyncio.to_thread(self._process_frame, frame, camera_params, tag_size, z_sign, camera_position)

    def _process_frame(self, frame, camera_params=None, tag_size=None, z_sign=1, camera_position=None):
        detections, dimg = self._get_detections(frame)
        poses = None
        spatial_pose_cubes = None
        distances = None
        rts = None
        if camera_params is not None and tag_size is not None:
            poses = self._detect_pose(detections, camera_params, tag_size)
            spatial_pose_cubes = []
            if len(poses) > 0:
                distances = []
                rts = []
                for index, pose_data in enumerate(poses):
                    if 'pose' in pose_data:
                        pose = pose_data['pose']
                        cube_lines, rt = self._calculate_cube_pose(pose, camera_params, tag_size, z_sign)
                        rts.append(rt)
                        spatial_pose_cubes.append(cube_lines)
                        # calculate distance from camera to cube front face
                        front_face_lines = cube_lines[:4]
                        # Extract unique points from front face lines to get rectangle corners
                        front_face_coords = []
                        for line in front_face_lines:
                            for point in line:
                                if point not in front_face_coords:
                                    front_face_coords.append(point)
                        distance = self._calc_distance_from_camera(pose, rt, camera_params, front_face_coords, tag_size)
                        distances.append(distance)
        return detections, poses, spatial_pose_cubes, distances, rts

    def _get_detections(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections, dimg = self.apriltag_detector.detect(gray, return_image=True)
        return detections, dimg

    def _calculate_cube_pose(self, pose, camera_params, tag_size, z_sign=1):
        results = []
        # opoints = np.array([ # cube
        #     -1, -1, 0,
        #     1, -1, 0,
        #     1, 1, 0,
        #     -1, 1, 0,
        #     -1, -1, -2 * z_sign,
        #     1, -1, -2 * z_sign,
        #     1, 1, -2 * z_sign,
        #     -1, 1, -2 * z_sign,
        # ]).reshape(-1, 1, 3) * 0.5 * tag_size
        opoints = np.array([ # square
            -1, -1, 0,
            1, -1, 0,
            1, 1, 0,
            -1, 1, 0
        ]).reshape(-1, 1, 3) * 0.5 * tag_size
        # edges = np.array([ # cube
        #     0, 1,
        #     1, 2,
        #     2, 3,
        #     3, 0,
        #     0, 4,
        #     1, 5,
        #     2, 6,
        #     3, 7,
        #     4, 5,
        #     5, 6,
        #     6, 7,
        #     7, 4
        # ]).reshape(-1, 2)
        edges = np.array([ # square
            0, 1,
            1, 2,
            2, 3,
            3, 0
        ]).reshape(-1, 2)
        fx, fy, cx, cy = camera_params
        K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
        rvec, _ = cv2.Rodrigues(pose[:3, :3])
        tvec = pose[:3, 3]
        dcoeffs = np.zeros(5)
        ipoints, _ = cv2.projectPoints(opoints, rvec, tvec, K, dcoeffs)
        ipoints = np.round(ipoints).astype(int)
        ipoints = [tuple(pt) for pt in ipoints.reshape(-1, 2)]
        for i, j in edges:
            results.append((ipoints[i], ipoints[j]))
        return results, (rvec, tvec)

    def _detect_pose(self, detections, camera_params, tag_size):
        results = []
        for detection in detections:
            pose, e0, e1 = self.apriltag_detector.detection_pose(detection, camera_params, tag_size)
            results.append({
                'pose': pose,
                'init_error': e0,
                'final_error': e1
            })
        return results

    def _calc_distance_from_camera(self, pose, rt, camera_params, front_face_coords, tag_size):
        tvec = pose[:3, 3]
        distance = np.linalg.norm(tvec)
        return distance

if __name__ == '__main__':
    from rich import print
    aw = ApriltagWorker()
    cap = cv2.VideoCapture(0)
    fx, fy, cx, cy = (np.float64(651.4091046043549), np.float64(653.2336134231156), np.float64(318.32692349087426), np.float64(234.8164852829485))
    camera_params = (fx, fy, cx, cy)
    # colors are in (blue, green, red) format
    # positions go clockwise, starting at the top for faces and top left for joining lines
    # see image apriltag_3d_cube_sample.png
    line_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), # front face
                   (0, 255, 255), (255, 0, 255), (128, 0, 128), (128, 128, 0), # lines joining front and back face
                   (0, 128, 128), (128, 0, 0), (0, 128, 0), (128, 128, 128)] # back face
    while cap.isOpened():
        ret, frame = cap.read()
        detections, poses, spatial_pose_cubes, distances, rts = aw._process_frame(frame, camera_params=camera_params, tag_size=100, z_sign=-1)
        if spatial_pose_cubes is not None and len(spatial_pose_cubes) > 0:
            for main_index, cube_lines in enumerate(spatial_pose_cubes):
                for index, line in enumerate(cube_lines):
                    cv2.line(frame, line[0], line[1], line_colors[index], 2)
                front_face_lines = cube_lines[:4]
                # Extract unique points from front face lines to get rectangle corners
                front_face_coords = []
                for line in front_face_lines:
                    for point in line:
                        if point not in front_face_coords:
                            front_face_coords.append(point)
                front_face_center_point = np.mean(front_face_coords, axis=0)
                cv2.circle(frame, (int(front_face_center_point[0]), int(front_face_center_point[1])), 5, (0, 0, 255), -1)
                distance = distances[main_index]
                rvec, tvec = rts[main_index]
                cv2.putText(frame, f'd = {distance:.2f}', (int(front_face_center_point[0]), int(front_face_center_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
