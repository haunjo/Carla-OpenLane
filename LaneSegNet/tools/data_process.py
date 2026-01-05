import numpy as np
from tqdm import tqdm
from shapely.geometry import LineString
from openlanev2.lanesegment.io import io

"""
This script is used to collect the data from the original OpenLane-V2 dataset.
The results will be saved in OpenLane-V2 folder.
The main difference between this script and the original one is that we don't interpolate the points for ped crossing and road bouadary.
"""

def _fix_pts_interpolate(curve, n_points):
    ls = LineString(curve)
    distances = np.linspace(0, ls.length, n_points)
    curve = np.array([ls.interpolate(distance).coords[0] for distance in distances], dtype=np.float32)
    return curve

def collect(root_path : str, data_dict : dict, collection : str, n_points : dict) -> None:

    data_list = [(split, segment_id, timestamp.split('.')[0]) \
        for split, segment_ids in data_dict.items() \
            for segment_id, timestamps in segment_ids.items() \
                for timestamp in timestamps
    ]
    meta = {}
    for split, segment_id, timestamp in tqdm(data_list, desc=f'collecting {collection}', ncols=100):
        identifier = (split, segment_id, timestamp)
        frame = io.json_load(f'{root_path}/{split}/{segment_id}/info/{timestamp}-ls.json')

        for k, v in frame['pose'].items():
            frame['pose'][k] = np.array(v, dtype=np.float64)
        for camera in frame['sensor'].keys():
            for para in ['intrinsic', 'extrinsic']:
                for k, v in frame['sensor'][camera][para].items():
                    frame['sensor'][camera][para][k] = np.array(v, dtype=np.float64)

        if 'annotation' not in frame:
            meta[identifier] = frame
            continue

        # NOTE: We don't interpolate the points for ped crossing and road bouadary.
        for i, area in enumerate(frame['annotation']['area']):
            frame['annotation']['area'][i]['points'] = np.array(area['points'], dtype=np.float32)

        # Collect text template labels for lanes (for text-guided topology reasoning)
        lane_text_labels = []
        for i, lane_segment in enumerate(frame['annotation']['lane_segment']):
            frame['annotation']['lane_segment'][i]['centerline'] = _fix_pts_interpolate(np.array(lane_segment['centerline']), n_points['centerline'])
            frame['annotation']['lane_segment'][i]['left_laneline'] = _fix_pts_interpolate(np.array(lane_segment['left_laneline']), n_points['left_laneline'])
            frame['annotation']['lane_segment'][i]['right_laneline'] = _fix_pts_interpolate(np.array(lane_segment['right_laneline']), n_points['right_laneline'])
            # Extract template_id if available (for Carla-OLV2 text-guided training)
            if 'template_id' in lane_segment:
                lane_text_labels.append(lane_segment['template_id'])

        # Add text labels as annotation field if any lanes have template_id
        if len(lane_text_labels) > 0:
            frame['annotation']['lane_text_labels'] = np.array(lane_text_labels, dtype=np.int8)

        for i, traffic_element in enumerate(frame['annotation']['traffic_element']):
            frame['annotation']['traffic_element'][i]['points'] = np.array(traffic_element['points'], dtype=np.float32)
        frame['annotation']['topology_lsls'] = np.array(frame['annotation']['topology_lsls'], dtype=np.int8)
        frame['annotation']['topology_lste'] = np.array(frame['annotation']['topology_lste'], dtype=np.int8)
        meta[identifier] = frame

    io.pickle_dump(f'{root_path}/{collection}.pkl', meta)

if __name__ == '__main__':
    root_path = 'data/Carla-OLV2/'

    # Process both train and val splits
    for split_name in ['train', 'val']:
        json_file = f'{root_path}/data_dict_carla_{split_name}_argoverse2.json'
        print(f"\n{'='*80}")
        print(f"Processing {split_name} split from {json_file}")
        print(f"{'='*80}\n")

        data_dict = io.json_load(json_file)

        # Generate pkl with proper naming
        collect(
            root_path,
            data_dict,
            f'data_dict_carla_argoverse2_{split_name}_lanesegnet',
            n_points={
                'centerline': 10,
                'left_laneline': 10,
                'right_laneline': 10
            },
        )
