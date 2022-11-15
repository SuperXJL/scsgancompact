import  argparse

def TrainSettings():
    parser = argparse.ArgumentParser()
    total_path = ['hit_stopwords.txt', 'cn_stopwords.txt', 'scu_stopwords.txt', 'baidu_stopwords.txt']
    # 数据地址
    parser.add_argument(
        '--data_dir', type=str, default=r'D:\我的项目\ProblemAboutCaching\RNN\tfRecords',
        help='Directory for storing input data')

    parser.add_argument(
        '--timeStep', type=int, default=5,
        help='timeStep')

    parser.add_argument(
        '--dimension', type=int, default=3,
        help='dimension')

    parser.add_argument(
        '--interval', type=int, default=1,
        help='interval')

    parser.add_argument(
        '--fOV_2DShape', type=int, default=[1024, 1024],
        help='fOV_2DShape')

    parser.add_argument(
        '--eyesight', type=int, default=2,
        help='eyesight')

    parser.add_argument(
        '--e_greedy', type=int, default=0.95,
        help='e_greedy')
    parser.add_argument(
        '--e_greedy_increment_c', type=int, default=0.00005,
        help='e_greedy_increment_c')
    parser.add_argument(
        '--gfu_bs_a', type=float, default=3.5,
        help='gfu_bs_a')
    parser.add_argument(
        '--irs_bs_a', type=float, default=2.5,
        help='f_mec')
    parser.add_argument(
        '--ue_irs_a', type=float, default=2.5,
        help='ue_irs_a')
    parser.add_argument(
        '--ue_bs_a', type=float, default=3.5,
        help='ue_bs_a')
    parser.add_argument(
        '--e_vr', type=float, default=10 ** (15),
        help='e_vr')
    parser.add_argument(
        '--r_min', type=int, default=1,
        help='r_min')
    parser.add_argument(
        '--fov_patch_num', type=int, default=64,
        help='fov_patch_num')
    parser.add_argument(
        '--BW', type=int, default=400,
        help='BW')
    parser.add_argument(
        '--training_interval', type=int, default=2,
        help='training_interval')
    parser.add_argument(
        '--double_q', type=bool, default=False,
        help='double_q')
    parser.add_argument(
        '--prioritized_r', type=bool, default=False,
        help='prioritized_r')

    parser.add_argument(
        '--replace_target_iter', type=float, default=50,
        help='replace_target_iter')
    parser.add_argument(
        '--antenna_num', type=int, default=8,
        help='antenna_num')
    parser.add_argument(
        '--bs_num', type=int, default=3,
        help='bs_num')
    parser.add_argument(
        '--ue_num', type=int, default=14,
        help='ue_num')
    parser.add_argument(
        '--irs_num', type=float, default=2,
        help='irs_num')
    parser.add_argument(
        '--p_max', type=float, default=80,
        help='p_max')
    parser.add_argument(
        '--irs_units_num', type=int, default=40,
        help='irs_units_num')
    parser.add_argument(
        '--memory_size', type=int, default=5000,
        help='irs_units_num')

    # model hyper-parameters
    # parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--compression_rate', type=float, default=0.1)
    # parser.add_argument('--num_classes', type=int, default=10)
    # VOC路径
    parser.add_argument('--train_root', type=str, default="../image_dataset/VOC/2007/train/JPEGImages")
    parser.add_argument('--val_root', type=str, default="../image_dataset/VOC/2007/test/JPEGImages")
    parser.add_argument('--test_root', type=str, default="../image_dataset/VOC/2007/test/JPEGImages")

    # COCO路径
    # parser.add_argument('--train_root', type=str, default="../image_dataset/data/coco/images/train2014")
    # parser.add_argument('--val_root', type=str, default="../image_dataset/data/coco/images/val2014")
    # parser.add_argument('--test_root', type=str, default="../image_dataset/data/coco/images/val2014")
    # training hyper-parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # misc
    parser.add_argument('--crop_w', type=int, default=64)  # 保持为2的n次方
    parser.add_argument('--crop_h', type=int, default=64)  # 保持为2的n次方
    parser.add_argument('--scale_rate', type=float, default=0.2)
    parser.add_argument('--state', type=str, default='train')
    parser.add_argument('--dataset', type=str, default='VOC', help="CoCo|VOC")
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--continue_learning', type=bool, default=False)
    parser.add_argument('--last_epoch', type=int, default=0)
    parser.add_argument('--model_path', type=str, default='./saved_models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--log_epoch', type=int, default=50)
    parser.add_argument('--sample_epoch', type=int, default=50)
    parser.add_argument('--validate_epoch', type=int, default=10)

    parser.add_argument('--input_channel', type=int, default=3)
    parser.add_argument('--output_channel', type=int, default=3)
    parser.add_argument('--first_kernel', type=int, default=5)
    parser.add_argument('--n_layers_D', type=int, default=3)

    parser.add_argument('--conv_out_channel', type=int, default=128)
    parser.add_argument('--conv_max_out_channel', type=int, default=512)


    # network saving and loading parameters

    parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--epoch_count', type=int, default=1,
                        help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

    # training parameters

    parser.add_argument('--n_epochs', type=int, default=5000, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=200,
                        help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
    parser.add_argument('--pool_size', type=int, default=50,

                        help='the size of image buffer that stores previously generated images')
    parser.add_argument('--lambda_L2', type=float, default=128, help='weights for the L2 loss')
    parser.add_argument('--lambda_feat', type=float, default=1, help='weights for the L2 loss')
    parser.add_argument('--is_Feat', action='store_true', default=1,
                        help='whether to use feature matching loss for generative training')
    parser.add_argument('--gpu_ids', type=int, default=[0],
                        help='indices of available cuda devices')  # [0,1]
    parser.add_argument('--is_feedback', type=bool, default=False,
                        help='is_feedback')
    parser.add_argument('--feedforward', type=str, default='IMPLICIT',
                        help='EXPLICIT-RES|IMPLICIT')
    parser.add_argument('--isTrain', type=bool, default=True,
                        help='isTrain')
    parser.add_argument('--lr_policy', type=str, default='linear',
                        help='learning rate policy. [linear | step | plateau | cosine]')                    

    parser.add_argument('--verbose', type=bool, default=True,
                        help='print the network architecture')
    parser.add_argument('--C_channel', type=int, default=8)
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--norm_EG', type=str, default="instance")
    parser.add_argument('--init_type', type=str, default='normal')
    parser.add_argument('--init_gain', type=float, default=0.02)
    parser.add_argument('--gan_mode', type=str, default='lsgan', help='[vanilla | lsgan | wgangp]')

    # OFDM parameters

    parser.add_argument('--P', type=int, default=16, help='number of packets for each transmitted image')
    parser.add_argument('--S', type=int, default=2, help='number of OFDM symbols per packet')
    parser.add_argument('--M', type=int, default=32, help='number of subcarriers per symbol')
    parser.add_argument('--K', type=int, default=16, help='length of cyclic prefix')
    parser.add_argument('--L', type=int, default=24, help='length of multipath channel')
    parser.add_argument('--decay', type=int, default=4, help='decay constant for the multipath channel')
    parser.add_argument('--is_clip', action='store_true', help='whether to include clipping')
    parser.add_argument('--CR', type=float, default=1.0, help='clipping ratio')
    parser.add_argument('--N_pilot', type=int, default=1, help='number of pilot symbols for channel estimation')
    parser.add_argument('--pilot', type=str, default='QPSK', help='type of pilots, choose from [QPSK | ZadoffChu]')
    parser.add_argument('--CE', type=str, default='LMMSE',
                        help='channel estimation method, choose from [LS | LMMSE | TRUE]')
    parser.add_argument('--EQ', type=str, default='MMSE', help='equalization method, choose from [ZF | MMSE]')
    parser.add_argument('--SNR', type=float, default=20.0, help='SNR')
    FLAGS, _ = parser.parse_known_args()

    return  parser


# model = rebuild_bert( batch_size=128, max_seq_length=20,
#                      is_training=True, categories=6, learning_rate=0.00005)
# # 加载要预测的数据
# model.load_data(x_train=x_train, x_test=x_test, vocab_file=vocab_file)
# # init_checkpoint = "D:\Chinese-BERT-wwm\\bert-use-demo\saved_models\chinese_L-12_H-768_A-12\\bert_model.ckpt"
# init_checkpoint = "D:\Chinese-BERT-wwm\\bert-use-demo\saved_models\model_iter380ac=0.7135416666666666.ckpt"
# model.train(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
#             init_checkpoint=init_checkpoint, iter_num=1000, iter_per_valid=20)
