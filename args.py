import argparse

def get_train_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num-visuals', type=int, default=10)
    parser.add_argument('--unif-importance', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ensemble-name', type=str, default='')
    parser.add_argument('--exp-crossreg', type=float, default=0)
    parser.add_argument('--save-dir', type=str, default='save/')
    parser.add_argument('--continue-dir', type=str, default='')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--train-datasets', type=str, default='squad,nat_questions,newsqa')
    parser.add_argument('--run-name', type=str, default='multitask_distilbert')
    parser.add_argument('--recompute-features', action='store_true')
    parser.add_argument('--train-dir', type=str, default='datasets/indomain_train')
    parser.add_argument('--val-dir', type=str, default='datasets/indomain_val')
    parser.add_argument('--eval-dir', type=str, default='datasets/oodomain_test')
    parser.add_argument('--eval-datasets', type=str, default='race,relation_extraction,duorc')
    parser.add_argument('--do-train', action='store_true')
    parser.add_argument('--do-eval', action='store_true')
    parser.add_argument('--sub-file', type=str, default='')
    parser.add_argument('--visualize-predictions', action='store_true')
    parser.add_argument('--eval-every', type=int, default=5000)
    parser.add_argument('--continue-train', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--num-experts',type=int, default=12)
    parser.add_argument('--num-init',type=int, default=0)
    parser.add_argument('--expert-size',type=int, default=1024)
    parser.add_argument('--force-serial',action='store_true')
    parser.add_argument('--visualize-worst',action='store_true')
    args = parser.parse_args()
    return args
