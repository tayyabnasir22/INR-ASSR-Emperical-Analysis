from Models.BaseTestingConfiguration import BaseTestingConfiguration
from Models.BenchmarkType import BenchmarkType
from Models.DecoderType import DecoderType
from Models.EncoderType import EncoderType
from TestingOrchestrators.BaseValidator import BaseValidator

class TestingOrchestrator:
    @staticmethod
    def TestDIV2K(encoder: EncoderType, decoder: DecoderType):
        scales = [2,3,4,6,12,18,24,30]

        configs = BaseTestingConfiguration()
        configs.encoder = encoder
        configs.decoder = decoder
        configs.benchmark = BenchmarkType.DIV2K
        configs.valid_data_path = './datasets/DIV2K/DIV2K_valid_HR'

        for scale in scales:
            print('Running DIV2K, Scale: ', str(scale), 'x')
            configs.eval_scale = scale
            if scale <= 4:
                configs.valid_data_pathScale = './datasets/DIV2K/DIV2K_valid_LRbicx' + str(scale)
            else:
                configs.valid_data_pathScale = None
            print('Configs: ', configs.to_dict())
            print('-'*33)
            BaseValidator().BuildAndValidate(
                **configs.to_dict()
            )
            print('-'*33)

    @staticmethod
    def TestSet5(encoder: EncoderType, decoder: DecoderType):
        scales = [2,3,4,6,8,12]

        configs = BaseTestingConfiguration()
        configs.encoder = encoder
        configs.decoder = decoder
        configs.benchmark = BenchmarkType.BENCHMARK
        configs.valid_data_path = './datasets/Set5/HR'

        for scale in scales:
            print('Running Set5, Scale: ', str(scale), 'x')
            configs.eval_scale = scale
            if scale <= 4:
                configs.valid_data_pathScale = './datasets/Set5/LR_bicubic/X' + str(scale)
            else:
                configs.valid_data_pathScale = None
            print('Configs: ', configs.to_dict())
            print('-'*33)
            BaseValidator().BuildAndValidate(
                **configs.to_dict()
            )
            print('-'*33)

    @staticmethod
    def TestSet14(encoder: EncoderType, decoder: DecoderType):
        scales = [2,3,4,6,8,12]

        configs = BaseTestingConfiguration()
        configs.encoder = encoder
        configs.decoder = decoder
        configs.benchmark = BenchmarkType.BENCHMARK
        configs.valid_data_path = './datasets/Set14/HR'

        for scale in scales:
            print('Running Set14, Scale: ', str(scale), 'x')
            configs.eval_scale = scale
            if scale <= 4:
                configs.valid_data_pathScale = './datasets/Set14/LR_bicubic/X' + str(scale)
            else:
                configs.valid_data_pathScale = None
            print('Configs: ', configs.to_dict())
            print('-'*33)
            BaseValidator().BuildAndValidate(
                **configs.to_dict()
            )
            print('-'*33)

    @staticmethod
    def TestB100(encoder: EncoderType, decoder: DecoderType):
        scales = [2,3,4,6,8,12]

        configs = BaseTestingConfiguration()
        configs.encoder = encoder
        configs.decoder = decoder
        configs.benchmark = BenchmarkType.BENCHMARK
        configs.valid_data_path = './datasets/B100/HR'

        for scale in scales:
            print('Running B100, Scale: ', str(scale), 'x')
            configs.eval_scale = scale
            if scale <= 4:
                configs.valid_data_pathScale = './datasets/B100/LR_bicubic/X' + str(scale)
            else:
                configs.valid_data_pathScale = None
            print('Configs: ', configs.to_dict())
            print('-'*33)
            BaseValidator().BuildAndValidate(
                **configs.to_dict()
            )
            print('-'*33)

    @staticmethod
    def TestUrban100(encoder: EncoderType, decoder: DecoderType):
        scales = [2,3,4,6,8,12]

        configs = BaseTestingConfiguration()
        configs.encoder = encoder
        configs.decoder = decoder
        configs.benchmark = BenchmarkType.BENCHMARK
        configs.valid_data_path = './datasets/Urban100/HR'

        for scale in scales:
            print('Running Urban100, Scale: ', str(scale), 'x')
            configs.eval_scale = scale
            if scale <= 4:
                configs.valid_data_pathScale = './datasets/Urban100/LR_bicubic/X' + str(scale)
            else:
                configs.valid_data_pathScale = None
            print('Configs: ', configs.to_dict())
            print('-'*33)
            BaseValidator().BuildAndValidate(
                **configs.to_dict()
            )
            print('-'*33)

    @staticmethod
    def TestCelebA_HQ(encoder: EncoderType, decoder: DecoderType):
        scales = [2,3,4,6,8,12]

        configs = BaseTestingConfiguration()
        configs.encoder = encoder
        configs.decoder = decoder
        configs.benchmark = BenchmarkType.BENCHMARK
        configs.valid_data_path = './datasets/CelebA-HQ/HR'

        for scale in scales:
            print('Running CelebA-HQ, Scale: ', str(scale), 'x')
            configs.eval_scale = scale
            
            configs.valid_data_pathScale = None
            print('Configs: ', configs.to_dict())
            print('-'*33)
            BaseValidator().BuildAndValidate(
                **configs.to_dict()
            )
            print('-'*33)

    @staticmethod
    def TestSVT(encoder: EncoderType, decoder: DecoderType):
        scales = [2,3,4,6,8,12]

        configs = BaseTestingConfiguration()
        configs.encoder = encoder
        configs.decoder = decoder
        configs.benchmark = BenchmarkType.BENCHMARK
        configs.valid_data_path = './datasets/SVT/HR'

        for scale in scales:
            print('Running SVT, Scale: ', str(scale), 'x')
            configs.eval_scale = scale
            
            configs.valid_data_pathScale = None
            print('Configs: ', configs.to_dict())
            print('-'*33)
            BaseValidator().BuildAndValidate(
                **configs.to_dict()
            )
            print('-'*33)

