import random
import torch.utils.data
from lib.utils import TensorDict
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms


def calc_mean_std(feat, eps=1e-5):
    C = feat.size(0)
    feat_var = feat.view(C, -1).var(dim=1) + eps
    feat_std = feat_var.sqrt().view(C, 1, 1)
    feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1, 1)
    return feat_mean, feat_std

def wallis(content, style, alpha=0):
    # assert (content.size() == style.size())
    style_mean, style_std = calc_mean_std(style)
    content_mean, content_std = calc_mean_std(content)
    normalized_feat = (content - content_mean.expand_as(content)) / content_std.expand_as(content)
    return ((normalized_feat * style_std.expand_as(content) + style_mean.expand_as(content)) * alpha + (1 - alpha) * content)

def wallis_cv2(content_image, style_image):
    content_tensor = transforms.ToTensor()(content_image)
    style_tensor = transforms.ToTensor()(style_image)

    output_tensor = wallis(content_tensor, style_tensor)

    output_image = output_tensor.cpu().clamp(0, 1).numpy()
    output_image = np.transpose(output_image, (1, 2, 0))
    output_image = (output_image * 255).astype(np.uint8)
    return output_image

def no_processing(data):
    return data


class TrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. 

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, day_datasets, night_datasets, p_day_datasets, p_night_datasets, samples_per_epoch, max_gap,
                 num_search_frames, num_template_frames=1, processing=no_processing, frame_sample_mode='causal',
                 train_cls=False, pos_prob=0.5):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.day_datasets = day_datasets
        self.night_datasets = night_datasets
        self.train_cls = train_cls  # whether we are training classification
        self.pos_prob = pos_prob  # probability of sampling positive class when making classification

        # If p not provided, sample uniformly from all videos
        if p_day_datasets is None:
            p_day_datasets = [len(d) for d in self.day_datasets]
        if p_night_datasets is None:
            p_night_datasets = [len(d) for d in self.night_datasets]
        # Normalize
        p_total = sum(p_day_datasets)
        self.p_day_datasets = [x / p_total for x in p_day_datasets]

        p_total = sum(p_night_datasets)
        self.p_night_datasets = [x / p_total for x in p_night_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,
                            allow_invisible=False, force_invisible=False):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        # get valid ids
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        return self.getitem()

    def getitem(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        while not valid:
            # Select a dataset
            dataset = random.choices(self.day_datasets, self.p_day_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)

            if is_video_dataset:
                template_frame_ids = None
                search_frame_ids = None
                gap_increase = 0

                while search_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                                max_id=len(visible) - self.num_search_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    template_frame_ids = base_frame_id + prev_frame_ids
                    search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                                max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                                num_ids=self.num_search_frames)
                    # Increase gap until a frame is found
                    gap_increase += 5
            else:
                raise ValueError("Only video dataset supported")
            try:
                day_template_frames, day_template_anno, _ = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
                day_search_frames, day_search_anno, _ = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

                H, W, _ = day_template_frames[0].shape
 
                day_data = TensorDict({'template_images': day_template_frames,
                                   'template_anno': day_template_anno['bbox'],
                                   'search_images': day_search_frames,
                                   'search_anno': day_search_anno['bbox']
                                })
                # make data augmentation
                
                day_data = self.processing(day_data)
                valid = day_data['valid']
                
            except:
                valid = False
        
        valid = False
        while not valid:
            # Select a dataset
            dataset = random.choices(self.night_datasets, self.p_night_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)

            if is_video_dataset:
                template_frame_ids = None
                search_frame_ids = None
                gap_increase = 0

                while search_frame_ids is None:
                        base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                                 max_id=len(visible) - self.num_search_frames, allow_invisible=True)
                        prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                  min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                  max_id=base_frame_id[0], allow_invisible=True)
                        if prev_frame_ids is None:
                            gap_increase += 5
                            continue
                        template_frame_ids = base_frame_id + prev_frame_ids
                        search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                                  max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                                  num_ids=self.num_search_frames, allow_invisible=True)
                        # Increase gap until a frame is found
                        gap_increase += 5
            else:
                raise ValueError("Only video dataset supported")
            try:
                night_template_frames, template_anno, _ = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
                night_search_frames, search_anno, _ = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

                H, W, _ = night_template_frames[0].shape
 
                night_data = TensorDict({'template_images': night_template_frames,
                                   'template_anno': template_anno['bbox'],
                                   'search_images': night_search_frames,
                                   'search_anno': search_anno['bbox']
                                })
                # make data augmentation
                
                night_data = self.processing(night_data)
                valid = night_data['valid']
                
            except:
                valid = False
        style_template_frames = [wallis_cv2(c,s) for c,s in zip(day_template_frames, night_template_frames)]
        style_search_frames = [wallis_cv2(c,s) for c,s in zip(day_search_frames, night_search_frames)]
        style_data = TensorDict({'template_images': style_template_frames,
                                'template_anno': day_template_anno['bbox'],
                                'search_images': style_search_frames,
                                'search_anno': day_search_anno['bbox']
                            })
        style_data = self.processing(style_data)

        data = TensorDict({
            'day_template_images': day_data['template_images'],
            'day_template_anno': day_data['template_anno'],
            'day_search_images': day_data['search_images'],
            'day_search_anno': day_data['search_anno'],
            'night_template_images': night_data['template_images'],
            'night_template_anno': night_data['template_anno'],
            'night_search_images': night_data['search_images'],
            'night_search_anno': night_data['search_anno'],
            'style_template_images': style_data['template_images'],
            'style_search_images': style_data['search_images']
        })
        
        return data

    def get_center_box(self, H, W, ratio=1/8):
        cx, cy, w, h = W/2, H/2, W * ratio, H * ratio
        return torch.tensor([int(cx-w/2), int(cy-h/2), int(w), int(h)])

    def sample_seq_from_dataset(self, dataset, is_video_dataset):

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_search_frames + self.num_template_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset
        return seq_id, visible, seq_info_dict