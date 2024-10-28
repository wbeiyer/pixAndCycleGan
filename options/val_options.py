from .base_options import BaseOptions


class ValOptions(BaseOptions):
    """This class includes val options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.num_threads = 0   # test code only supports num_threads = 0
        parser.batch_size = 1    # test code only supports batch_size = 1
        parser.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        parser.no_flip = True    # no flip; comment this line if results on flipped images are needed.
        parser.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

        parser.add_argument('--validation_freq', type=int, default=10, help='val frequence')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--Ftest', type=int, default=100000, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
