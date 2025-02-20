import numpy as np
import torch

from qpid.constant import INPUT_TYPES
from qpid.model import Model, layers, process
from qpid.utils import get_mask

from .__args import GrouplusModelArgs

NORMALIZED_SIZE = None

INF = 100000000
SAFE_THRESHOLDS = 0.05
MU = 0.00001


class ConceptionLayer(torch.nn.Module):
    """
    A layer to compute social interaction around target agent.
    """

    def __init__(self,
                 use_view_angle: bool | int = True,
                 view_angle: float = np.pi,
                 use_pooling: bool | int = True,
                 use_max: bool | int = False,
                 use_velocity: bool | int = True,
                 use_distance: bool | int = True,
                 use_move_dir: bool | int = True,
                 use_group: bool | int = True,
                 disable_conception: bool | int = False,
                 *args, **kwargs):
        """
        ## View Angle Settings
        :param use_view_angle: Whether to consider view angle of the target agent.
        :param view_angle: View angle of the target agent.
        :param use_pooling: Choose whether to use pooling in calculating conception value. Only choose one between pooling and max.
        :param use_max: Choose whether to use max in calculating conception value. Only choose one between pooling and max.
        :param use_velocity: Choose whether to use the velocity factor in the conception.
        :param use_distance: Choose whether to use the distance factor in the conception.
        :param use_move_dir: Choose whether to use the move direction factor in the conception.
        :param use_group: Choose whether to use pedestrian groups when calculating SocialCircle.
        :param disable_conception: Choose whether to disable conception layer in the GroupModel.
        """
        super().__init__(*args, **kwargs)
        self.use_view_angle = use_view_angle
        self.view_angle = view_angle
        self.use_pooling = use_pooling
        self.use_max = use_max
        self.use_velocity = use_velocity
        self.use_distance = use_distance
        self.use_move_dir = use_move_dir
        self.use_group = use_group
        self.diable_conception = disable_conception

    @property
    def dim(self) -> int:
        """
        The number of conception layer factors.
        """
        return (self.use_velocity + self.use_distance + self.use_move_dir)

    def forward(self, trajs, nei_trajs, *args, **kwargs):
        # `nei_trajs` are relative values to target agents' last obs step
        obs_vector = trajs[..., -1:, :] - trajs[..., 0:1, :]
        nei_vector = nei_trajs[..., -1, :] - nei_trajs[..., 0, :]
        nei_posion_vector = nei_trajs[..., -1, :]

        # obs's direction is simplified to be its moving direction during the last interval
        obs_dir_vec = trajs[..., -1:, :] - trajs[..., -2:-1, :]
        obs_dir = torch.atan2(obs_dir_vec[..., 0], obs_dir_vec[..., 1])
        obs_dir = obs_dir % (2*np.pi)

        # neighbor's direction
        nei_dir = torch.atan2(nei_posion_vector[..., 0],
                              nei_posion_vector[..., 1])
        nei_dir = nei_dir % (2*np.pi)

        # mask neighbors
        nei_mask = (
            torch.sum(nei_trajs, dim=[-1, -2]) < (0.05 * INF)).to(dtype=torch.int32)

        # mask view angle
        view_mask = (torch.abs(nei_dir - obs_dir) <
                     (self.view_angle / 2)).to(dtype=torch.int32)
        left_view_mask = ((nei_dir - obs_dir) > 0).to(dtype=torch.int32)
        right_view_mask = view_mask - left_view_mask

        # mask back angle(places out of the view)
        back_mask = 1 - view_mask

        # all real neighbors in left view, right view and back view
        nei_left = left_view_mask * nei_mask
        nei_right = right_view_mask * nei_mask
        nei_view = nei_left + nei_right
        nei_back = back_mask * nei_mask

        # calculate neighbors' distance
        dis = torch.norm(nei_posion_vector, dim=-1)

        # calculate neighbors' moving direction
        nei_move_dir_vec = nei_trajs[..., -1:, :] - nei_trajs[..., -2:-1, :]
        nei_move_dir = torch.atan2(
            nei_move_dir_vec[..., 0], nei_move_dir_vec[..., 1])
        nei_move_dir = nei_move_dir % (2*np.pi)
        delta_dir = torch.squeeze(
            (nei_move_dir - obs_dir[:, None, ...]), dim=-1)

        # calculate neighbor's velocity
        velocity = torch.norm(nei_vector, dim=-1)

        # for neighbors in view angle, the conception layer would consider all three factors
        # for neighbors in the back, the conception layer would only consider distance factor
        if self.use_pooling:

            # calculate conception value in right view
            dis_right = (torch.sum(dis * nei_right,
                         dim=[-1, -2])) / (torch.sum(nei_right, dim=-1) + MU)
            dir_right = (torch.sum(delta_dir * nei_right,
                         dim=[-1, -2])) / (torch.sum(nei_right, dim=-1) + MU)
            vel_right = (torch.sum(velocity * nei_right,
                         dim=[-1, -2])) / (torch.sum(nei_right, dim=-1) + MU)
            con_right = torch.concat(
                [dis_right[:, None, None], dir_right[:, None, None], vel_right[:, None, None]], dim=-1)

            # calculate conception value in left view
            dis_left = (torch.sum(dis * nei_left,
                        dim=[-1, -2])) / (torch.sum(nei_left, dim=-1) + MU)
            dir_left = (torch.sum(delta_dir * nei_left,
                        dim=[-1, -2])) / (torch.sum(nei_left, dim=-1) + MU)
            vel_left = (torch.sum(velocity * nei_left,
                        dim=[-1, -2])) / (torch.sum(nei_left, dim=-1) + MU)
            con_left = torch.concat(
                [dis_left[:, None, None], dir_left[:, None, None], vel_left[:, None, None]], dim=-1)

            # calculate conception in the back
            dis_back = (torch.sum(dis * nei_back,
                        dim=[-1, -2])) / (torch.sum(nei_back, dim=-1) + MU)
            con_back = torch.concat([dis_back[:, None, None]], dim=-1) 
            # / self.dim, dis_back[:,
            #                         None, None] / self.dim, dis_back[:, None, None] / self.dim], dim=-1)

            # add right and left
            con = torch.concat([con_right, con_left, con_back], dim=-1)

            # -----------------------
            # The following lines are used to draw visualized figures in our paper
            # from scripts.draw_fov import draw_fov

            # start_angle = (obs_dir - (self.view_angle/2)) * (180/(np.pi))
            # start_angle = start_angle[0].item()
            # p = torch.concat([torch.sum(con_right, dim=-1), torch.sum(con_left, dim=-1), torch.sum(con_back, dim=-1)], dim=-1)

            # draw_fov(con=p, file_name='conception_left_right', color_high=[0xff, 0xa6, 0x00],
            #             color_low=[0x00, 0x3f, 0x5c],
            #             max_width=0.3, min_width=0.2, view_angle=self.view_angle*(180/(np.pi)),
            #             start_angle=start_angle)
            # Vis codes end here
            # -----------------------

            return con

        else:
            # calculate conception value in right view
            dis_right = (torch.max(dis * nei_right))
            dir_right = (torch.max(delta_dir * nei_right))
            vel_right = (torch.max(velocity * nei_right))
            con_right = torch.concat(
                [dis_right[:, None, None], dir_right[:, None, None], vel_right[:, None, None]], dim=-1)

            # calculate conception value in left view
            dis_left = (torch.max(dis * nei_left))
            dir_left = (torch.max(delta_dir * nei_left))
            vel_left = (torch.max(velocity * nei_left))
            con_left = torch.concat(
                [dis_left[:, None, None], dir_left[:, None, None], vel_left[:, None, None]], dim=-1)

            # calculate conception in the back
            dis_back = (torch.max(dis * nei_back))
            con_back = torch.concat([dis_back[:, None, None] / self.dim, dis_back[:,
                                    None, None] / self.dim, dis_back[:, None, None] / self.dim], dim=-1)

            # add right and left
            con = torch.concat([con_right, con_left, con_back], dim=-2)

            return con

    def implement(self, model: Model, inputs: list[torch.Tensor]):
        obs = model.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        nei = model.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)
        if (self.use_group and (not self.diable_conception)):
            # Long term distance between neighbors and obs
            long_term_dis = nei - obs[:, None, ...]# final step distance(fde)
            final_vec = nei[..., -1:, :] - obs[:, None, -1:, :]
            group_mask = ((torch.sum(long_term_dis ** 2,
                                    dim=[-1, -2]) < 6).to(dtype=torch.int32)) * ((torch.sum(final_vec ** 2, dim=[-1, -2]) < 6/8).to(dtype=torch.int32))
            trajs_group = nei * group_mask[..., None, None]
            nei_trajs = nei * \
                (1 - group_mask[..., None, None]) + \
                group_mask[..., None, None] * INF
            con = self(obs, nei_trajs)

            return con

        elif (self.use_group and self.diable_conception):
            nei_trajs = torch.ones_like(nei) * INF
            con = self(obs, nei_trajs)

            return con

        else:
            con = self(obs, nei)

            return con


class SegmapLayer(torch.nn.Module):
    """
    A layer to compute interactions between target agent and surroundings.
    """
    def __init__(self,
                 use_view_angle: bool | int = True,
                 view_angle: float = np.pi,
                 use_pooling: bool | int = True,
                 use_max: bool | int = False,
                 use_velocity: bool | int = True,
                 use_distance: bool | int = True,
                 use_move_dir: bool | int = True,
                 use_group: bool | int = True,
                 disable_conception: bool | int = False,
                 vision_radius: float = 2.0,
                 pool_size: int = -1,
                 *args, **kwargs):
        """
        ## View Angle Settings
        :param use_view_angle: Whether to consider view angle of the target agent.
        :param view_angle: View angle of the target agent.
        :param use_pooling: Choose whether to use pooling in calculating conception value. Only choose one between pooling and max.
        :param use_max: Choose whether to use max in calculating conception value. Only choose one between pooling and max.
        :param use_velocity: Choose whether to use the velocity factor in the conception.
        :param use_distance: Choose whether to use the distance factor in the conception.
        :param use_move_dir: Choose whether to use the move direction factor in the conception.
        :param use_group: Choose whether to use pedestrian groups when calculating SocialCircle.
        :param disable_conception: Choose whether to disable conception layer in the GroupModel.
        """
        super().__init__(*args, **kwargs)

        from qpid.mods.segMaps.settings import NORMALIZED_SIZE

        self.use_view_angle = use_view_angle
        self.view_angle = view_angle
        self.use_pooling = use_pooling
        self.use_max = use_max
        self.use_velocity = use_velocity
        self.use_distance = use_distance
        self.use_move_dir = use_move_dir
        self.use_group = use_group
        self.diable_conception = disable_conception
        self.radius = vision_radius

        # Compute all pixels' indices
        pool_size = 1 if pool_size == -1 else pool_size
        xs, ys = torch.meshgrid(torch.arange(NORMALIZED_SIZE//pool_size),
                                torch.arange(NORMALIZED_SIZE//pool_size),
                                indexing='ij')
        self.map_pos_pixel = torch.stack(
            [xs.reshape([-1]), ys.reshape([-1])], dim=-1).to(torch.float32)
        self.map_pos_pixel = self.map_pos_pixel * pool_size + pool_size // 2

        if pool_size > 1:
            self.pool = torch.nn.MaxPool2d((pool_size, pool_size))
        else:
            self.pool = None

    @property
    def dim(self) -> int:
        """
        The number of conception layer factors.
        """
        return (self.use_velocity + self.use_distance + self.use_move_dir)
    
    def forward(self, seg_maps: torch.Tensor,
                seg_map_paras: torch.Tensor,
                trajectories: torch.Tensor,
                current_pos: torch.Tensor,
                *args, **kwargs):
        
        # Move back to original trajectories
        _obs = trajectories + current_pos

        # Treat seg maps as a long sequence
        if self.pool:
            _maps = self.pool(seg_maps[..., None, :, :])[..., 0, :, :]
        else:
            _maps = seg_maps

        _maps = torch.flatten(_maps, start_dim=1, end_dim=-1)
        map_safe_mask = (_maps <= SAFE_THRESHOLDS).to(torch.float32)

        # Compute velocity (moving length) during observation period
        moving_vector = _obs[..., -1, :] - _obs[..., 0, :]
        moving_length = torch.norm(moving_vector, dim=-1)   # (batch)

        # Compute pixel positions on seg maps
        W = seg_map_paras[..., :2][..., None, :]
        b = seg_map_paras[..., 2:4][..., None, :]

        # Compute angles and distances
        self.map_pos_pixel = self.map_pos_pixel.to(W.device)
        map_pos = (self.map_pos_pixel - b) / W  # (batch, a*a, 2)

        # Compute distances and angles of all pixels
        direction_vectors = map_pos - current_pos           # (batch, a*a, 2)
        dis = torch.norm(direction_vectors, dim=-1)   # (batch, a*a)

        angles = torch.atan2(direction_vectors[..., 0],
                             direction_vectors[..., 1])     # (batch, a*a)
        
        # Compute the `equivalent` distance
        equ_dis = (dis + MU) / (_maps + MU)

        # Compute the vision range
        r = (self.radius * moving_length)[..., None]
        radius_mask = (dis <= r).to(torch.float32)

        # # `nei_trajs` are relative values to target agents' last obs step
        # obs_vector = trajs[..., -1:, :] - trajs[..., 0:1, :]
        # nei_vector = nei_trajs[..., -1, :] - nei_trajs[..., 0, :]
        # nei_posion_vector = nei_trajs[..., -1, :]

        # obs's direction is simplified to be its moving direction during the last interval
        obs_dir_vec = _obs[..., -1:, :] - _obs[..., -2:-1, :]
        obs_dir = torch.atan2(obs_dir_vec[..., 0], obs_dir_vec[..., 1])
        obs_dir = obs_dir % (2*np.pi)

        # neighbor's direction
        nei_dir = angles % (2*np.pi)
        # nei_dir = torch.atan2(nei_posion_vector[..., 0],
        #                       nei_posion_vector[..., 1])
        # nei_dir = nei_dir % (2*np.pi)

        # mask neighbors
        nei_mask = (
            torch.sum(map_pos, dim=-1) < (0.05 * INF)).to(dtype=torch.int32)
        
        # mask vision_radius
        nei_mask = nei_mask * radius_mask

        # mask view angle
        view_mask = (torch.abs(nei_dir - obs_dir) <
                     (self.view_angle / 2)).to(dtype=torch.int32)
        left_view_mask = ((nei_dir - obs_dir) > 0).to(dtype=torch.int32)
        right_view_mask = view_mask - left_view_mask

        # mask back angle(places out of the view)
        back_mask = 1 - view_mask

        # all real neighbors in left view, right view and back view
        nei_left = left_view_mask * nei_mask
        nei_right = right_view_mask * nei_mask
        nei_view = nei_left + nei_right
        nei_back = back_mask * nei_mask

        # # calculate neighbors' distance
        # dis = torch.norm(nei_posion_vector, dim=-1)

        # calculate neighbors' moving direction
        nei_move_dir_vec = _obs[..., -2:-1, :] - _obs[..., -1:, :] 
        nei_move_dir = torch.atan2(
            nei_move_dir_vec[..., 0], nei_move_dir_vec[..., 1])
        nei_move_dir = nei_move_dir % (2*np.pi)
        delta_dir = nei_move_dir.repeat(1, 10000) - obs_dir

        # calculate neighbor's velocity
        nei_vector = _obs[..., -1:, :] - _obs[..., 0:1, :]
        velocity = torch.norm(nei_vector, dim=-1)

        # for neighbors in view angle, the conception layer would consider all three factors
        # for neighbors in the back, the conception layer would only consider distance factor
        if self.use_pooling:

            # calculate conception value in right view
            dis_right = (torch.sum(dis * nei_right,
                         dim=[-1, -2])) / (torch.sum(nei_right, dim=-1) + MU)
            dir_right = (torch.sum(delta_dir * nei_right,
                         dim=[-1, -2])) / (torch.sum(nei_right, dim=-1) + MU)
            vel_right = (torch.sum(velocity * nei_right,
                         dim=[-1, -2])) / (torch.sum(nei_right, dim=-1) + MU)
            con_right = torch.concat(
                [dis_right[:, None, None], dir_right[:, None, None], vel_right[:, None, None]], dim=-1)

            # calculate conception value in left view
            dis_left = (torch.sum(dis * nei_left,
                        dim=[-1, -2])) / (torch.sum(nei_left, dim=-1) + MU)
            dir_left = (torch.sum(delta_dir * nei_left,
                        dim=[-1, -2])) / (torch.sum(nei_left, dim=-1) + MU)
            vel_left = (torch.sum(velocity * nei_left,
                        dim=[-1, -2])) / (torch.sum(nei_left, dim=-1) + MU)
            con_left = torch.concat(
                [dis_left[:, None, None], dir_left[:, None, None], vel_left[:, None, None]], dim=-1)

            # calculate conception in the back
            dis_back = (torch.sum(dis * nei_back,
                        dim=[-1, -2])) / (torch.sum(nei_back, dim=-1) + MU)
            con_back = torch.concat([dis_back[:, None, None]], dim=-1) 
            # / self.dim, dis_back[:,
            #                         None, None] / self.dim, dis_back[:, None, None] / self.dim], dim=-1)

            # add right and left
            con = torch.concat([con_right, con_left, con_back], dim=-1)

            # -----------------------
            # The following lines are used to draw visualized figures in our paper
            # from scripts.draw_fov import draw_fov

            # start_angle = (obs_dir - (self.view_angle/2)) * (180/(np.pi))
            # start_angle = start_angle[0].item()
            # p = torch.concat([torch.sum(con_right, dim=-1), torch.sum(con_left, dim=-1), torch.sum(con_back, dim=-1)], dim=-1)

            # draw_fov(con=p, file_name='conception_left_right', color_high=[0xff, 0xa6, 0x00],
            #             color_low=[0x00, 0x3f, 0x5c],
            #             max_width=0.3, min_width=0.2, view_angle=self.view_angle*(180/(np.pi)),
            #             start_angle=start_angle)
            # Vis codes end here
            # -----------------------

            return con

        else:
            # calculate conception value in right view
            dis_right = (torch.max(dis * nei_right))
            dir_right = (torch.max(delta_dir * nei_right))
            vel_right = (torch.max(velocity * nei_right))
            con_right = torch.concat(
                [dis_right[:, None, None], dir_right[:, None, None], vel_right[:, None, None]], dim=-1)

            # calculate conception value in left view
            dis_left = (torch.max(dis * nei_left))
            dir_left = (torch.max(delta_dir * nei_left))
            vel_left = (torch.max(velocity * nei_left))
            con_left = torch.concat(
                [dis_left[:, None, None], dir_left[:, None, None], vel_left[:, None, None]], dim=-1)

            # calculate conception in the back
            dis_back = (torch.max(dis * nei_back))
            con_back = torch.concat([dis_back[:, None, None] / self.dim, dis_back[:,
                                    None, None] / self.dim, dis_back[:, None, None] / self.dim], dim=-1)

            # add right and left
            con = torch.concat([con_right, con_left, con_back], dim=-2)

            return con

    def implement(self, model: Model, inputs: list[torch.Tensor]):
        
        from qpid.mods import segMaps

        obs = model.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        
         # Segmentaion-map-related inputs (to compute the PhysicalCircle)
        # (batch, h, w)
        seg_maps = model.get_input(inputs, segMaps.INPUT_TYPES.SEG_MAP)

        # (batch, 4)
        seg_map_paras = model.get_input(
            inputs, segMaps.INPUT_TYPES.SEG_MAP_PARAS)
        
        # Process model inputs
        sl_args = model.args.register_subargs(GrouplusModelArgs, 'sl')
        if sl_args.use_empty_segmap:
            seg_maps = torch.zeros_like(seg_maps)

        if (m_layer := model.processor.get_layer_by_type(process.Move)):
            unprocessed_pos = m_layer.ref_points
        else:
            unprocessed_pos = torch.zeros_like(obs[..., -1:, :])

        # Start computing the PhysicalCircle
        # PhysicalCircle will be computed on each agent's 2D center point
        c_obs = model.picker.get_center(obs)[..., :2]
        c_unpro_pos = model.picker.get_center(unprocessed_pos)[..., :2]

        # Compute PhysicalCircle meta components
        seg_conception = self(seg_maps, seg_map_paras, c_obs, c_unpro_pos)

        # Rotate the PhysicalCircle (if needed)
        # if (r_layer := model.processor.get_layer_by_type(process.Rotate)):
        #     seg_conception = self.rotate(seg_conception, r_layer.angles)

        return seg_conception

    # def rotate(self, circle: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    #     """
    #     Rotate the physicalCircle. (Usually used after preprocess operations.)
    #     """
    #     # Rotate the circle <=> left or right shift the circle
    #     # Compute shift length
    #     angles = angles % (2*np.pi)
    #     partition_angle = (2*np.pi) / (self.partitions)
    #     move_length = (angles // partition_angle).to(torch.int32)

    #     # Remove paddings
    #     valid_circle = circle[..., :self.partitions, :]
    #     valid_circle = torch.concat([valid_circle, valid_circle], dim=-2)
    #     paddings = circle[..., self.partitions:, :]

    #     # Shift each circle
    #     rotated_circles = []
    #     for _circle, _move in zip(valid_circle, move_length):
    #         rotated_circles.append(_circle[_move:self.partitions+_move])

    #     rotated_circles = torch.stack(rotated_circles, dim=0)
    #     return torch.concat([rotated_circles, paddings], dim=-2)
    

class FusionLayer(torch.nn.Module):

    def __init__(self, feature_dimension=128, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.d = feature_dimension
        self.fc1 = layers.Dense(7, self.d, torch.nn.Tanh)
        self.fc2 = layers.Dense(self.d, 1, torch.nn.Sigmoid)

    def forward(self, con: torch.Tensor, seg: torch.Tensor, *args, **kwargs):

        w_partition_sc = self.fc2(self.fc1(con))     # (batch, p, 1)
            
        w_partition_pc = self.fc2(self.fc1(seg))     # (batch, p, 1)

        return ((w_partition_sc * con + w_partition_pc * seg) /
                    (w_partition_sc + w_partition_pc + MU))