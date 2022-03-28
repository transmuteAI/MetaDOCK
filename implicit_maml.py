import numpy as np
import pandas as pd
import os
import torch
import argparse
import sys
import utils
import pickle5 as pickle


from learner_model import Learner, make_conv_network
from utils import seed_all, StaticSampler


from torchmeta.datasets.helpers import cifar_fs, miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader

# ===================
# hyperparameters
# ===================
# 
parser = argparse.ArgumentParser(description='Implicit MAML')
parser.add_argument('--dataset', type=str, default='cifar_fs', help='One of - cifar_fs, miniimagenet')
parser.add_argument('--model', type=str, default='4conv_1', help='4conv_1 / 4conv_2')
parser.add_argument('--N_way', type=int, default=5, help='number of classes for few shot learning tasks')
parser.add_argument('--K_shot', type=int, default=5, help='number of instances for few shot learning tasks')
parser.add_argument('--inner_lr', type=float, default=5e-3, help='inner loop learning rate')
parser.add_argument('--outer_lr', type=float, default=5e-5, help='outer loop learning rate')
parser.add_argument('--n_steps', type=int, default=16, help='number of steps in inner loop')
parser.add_argument('--n_steps_val', type=int, default=16, help='number of steps in inner loop')
parser.add_argument('--meta_steps', type=int, default=60000, help='number of meta steps')
parser.add_argument('--task_mb_size', type=int, default=4)
parser.add_argument('--lam', type=float, default=0.5, help='regularization in inner steps')
parser.add_argument('--cg_steps', type=int, default=5)
parser.add_argument('--cg_damping', type=float, default=1.0)
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--save_dir', type=str, default='exps/pretraining')
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--val_tasks', type=int, default=600)
parser.add_argument('--val_every', type=int, default=1000)
parser.add_argument('--val_tasks_path', type=str, default='tasks/num_tasks_600_num_classes_16_ways_5.pkl')
parser.add_argument('--train', type=int, default=1)
parser.add_argument('--test', type=int, default=1)
parser.add_argument('--test_val', type=int, default=0)
args = parser.parse_args()

def main():
    seed_all(args.seed)

    train_dataset = globals()[args.dataset]("data", ways=args.N_way, shots=args.K_shot, test_shots=args.K_shot, meta_train=True, download=True)
    train_dataloader = BatchMetaDataLoader(train_dataset, batch_size=args.task_mb_size, num_workers=args.workers)

    learner_net = make_conv_network(out_dim=args.N_way, task=args.dataset, model_name=args.model, method = 'full')
    fast_net = make_conv_network(out_dim=args.N_way, task=args.dataset, model_name=args.model, method = 'full')
    meta_learner = Learner(model=learner_net, loss_function=torch.nn.CrossEntropyLoss(),
                        inner_lr=args.inner_lr, outer_lr=args.outer_lr, inner_alg='gradient', GPU=args.use_gpu)
    fast_learner = Learner(model=fast_net, loss_function=torch.nn.CrossEntropyLoss(),
                        inner_lr=args.inner_lr, outer_lr=args.outer_lr, inner_alg='gradient', GPU=args.use_gpu)
    meta_learner.hist['loss'] = np.zeros((args.meta_steps, 4))
    meta_learner.hist['accuracy'] = np.zeros((args.meta_steps, 2))
    meta_learner.hist['val_accuracy'] = np.zeros((args.meta_steps//args.val_every, 2))

    device = 'cuda' if args.use_gpu is True else 'cpu'
    lam = torch.tensor(args.lam)
    lam = lam.to(device)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(args.save_dir+'/flagfile.txt', 'w') as f:
        f.write('\n'.join(sys.argv[1:]))

    if args.train:
        # ===================
        # Train
        # ===================
        print("Training model ......")
        for outstep, batch in enumerate(train_dataloader, start=len(meta_learner.hist['loss']) - args.meta_steps):
            if outstep==len(meta_learner.hist['loss']):
                break
            w_k = meta_learner.get_params()
            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=device)
            train_targets = train_targets.to(device=device)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=device)
            test_targets = test_targets.to(device=device)
            meta_grad = 0.0
            lam_grad = 0.0

            for idx in range(args.task_mb_size):
                fast_learner.set_params(w_k.clone())

                vl_before = fast_learner.get_loss(test_inputs[idx], test_targets[idx], return_numpy=True)
                tl = fast_learner.learn_task(train_inputs[idx], train_targets[idx], num_steps=args.n_steps)
                # pull back for regularization
                fast_learner.inner_opt.zero_grad()
                regu_loss = fast_learner.regularization_loss(w_k, lam)
                regu_loss.backward()
                fast_learner.inner_opt.step()

                tacc = utils.measure_accuracy(train_inputs[idx], train_targets[idx], fast_learner)
                vacc = utils.measure_accuracy(test_inputs[idx], test_targets[idx], fast_learner)

                valid_loss = fast_learner.get_loss(test_inputs[idx], test_targets[idx])
                vl_after = utils.to_numpy(valid_loss).ravel()[0]
                valid_grad = torch.autograd.grad(valid_loss, fast_learner.model.parameters())
                flat_grad = torch.cat([g.contiguous().view(-1) for g in valid_grad])

                if args.cg_steps <= 1:
                    task_outer_grad = flat_grad
                else:
                    task_matrix_evaluator = fast_learner.matrix_evaluator(train_inputs[idx], train_targets[idx], lam, args.cg_damping)
                    task_outer_grad = utils.cg_solve(task_matrix_evaluator, flat_grad, args.cg_steps, x_init=None)

                meta_grad += (task_outer_grad/args.task_mb_size)
                meta_learner.hist['loss'][outstep] += (np.array([tl[0], vl_before, tl[-1], vl_after])/args.task_mb_size)
                meta_learner.hist['accuracy'][outstep] += np.array([tacc, vacc]) / args.task_mb_size

            meta_learner.outer_step_with_grad(meta_grad, flat_grad=True)

            if (outstep % args.val_every == 0 and outstep > 0) or outstep==args.meta_steps-1:
                loss = pd.DataFrame(meta_learner.hist['loss'], columns = ['train_pre', 'test_pre', 'train_post', 'test_post'])
                accuracy = pd.DataFrame(meta_learner.hist['accuracy'], columns = ['train_acc', 'val_acc'])
                loss.to_csv(args.save_dir + '/loss.csv')
                accuracy.to_csv(args.save_dir + '/accuracy.csv')
            if (outstep % args.val_every == 0 and outstep > 0) or outstep==args.meta_steps-1:
                print("Validating model ......")
                sampler = StaticSampler(args.val_tasks_path)
                val_dataset = globals()[args.dataset]("data", ways=args.N_way, shots=args.K_shot, test_shots=args.K_shot, meta_test=True, download=True)
                val_dataloader = BatchMetaDataLoader(val_dataset, batch_size=args.task_mb_size, num_workers=args.workers, sampler=sampler, shuffle=False)
                losses = np.zeros((args.val_tasks, 4))
                accuracy = np.zeros((args.val_tasks, 2))
                meta_params = meta_learner.get_params().clone()

                for i, batch in enumerate(val_dataloader):
                    if i==args.val_tasks/args.task_mb_size:
                        break
                    train_inputs, train_targets = batch['train']
                    train_inputs = train_inputs.to(device=device)
                    train_targets = train_targets.to(device=device)

                    test_inputs, test_targets = batch['test']
                    test_inputs = test_inputs.to(device=device)
                    test_targets = test_targets.to(device=device)

                    for idx in range(args.task_mb_size):
                        fast_learner.set_params(meta_params) # sync weights
                        vl_before = fast_learner.get_loss(test_inputs[idx], test_targets[idx], return_numpy=True)
                        tl = fast_learner.learn_task(train_inputs[idx], train_targets[idx], num_steps=args.n_steps_val, add_regularization=False)
                        fast_learner.inner_opt.zero_grad()
                        regu_loss = fast_learner.regularization_loss(meta_params, args.lam)
                        regu_loss.backward()
                        fast_learner.inner_opt.step()
                        vl_after = fast_learner.get_loss(test_inputs[idx], test_targets[idx], return_numpy=True)
                        tacc = utils.measure_accuracy(train_inputs[idx], train_targets[idx], fast_learner)
                        vacc = utils.measure_accuracy(test_inputs[idx], test_targets[idx], fast_learner)
                        losses[i*args.task_mb_size+idx] = np.array([tl[0], vl_before, tl[-1], vl_after])
                        accuracy[i*args.task_mb_size+idx][0] = tacc; accuracy[i*args.task_mb_size+idx][1] = vacc

                if np.mean(accuracy, axis=0)[1] >= meta_learner.hist['val_accuracy'][:,0].max():
                    checkpoint_file = args.save_dir+'/final_model.pickle'
                    pickle.dump(meta_learner, open(checkpoint_file, 'wb'), pickle.HIGHEST_PROTOCOL)
                meta_learner.hist['val_accuracy'][outstep//args.val_every][0] = np.mean(accuracy, axis=0)[1]
                meta_learner.hist['val_accuracy'][outstep//args.val_every][1] = 1.96*np.std(accuracy, axis=0)[1]/np.sqrt(args.val_tasks)
                val_accuracy = pd.DataFrame(meta_learner.hist['val_accuracy'], columns = ['val_acc', 'val_std'])
                val_accuracy.to_csv(args.save_dir + '/val_accuracy.csv')
                
    if args.test:
        print("Testing model ......")
        sampler = StaticSampler(args.val_tasks_path)
        test_dataset = globals()[args.dataset]("data", ways=args.N_way, shots=args.K_shot, test_shots=args.K_shot, meta_test=True, download=True)
        if not args.test_val:
            test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=args.task_mb_size, num_workers=args.workers, shuffle=False)
            test_tasks = len(test_dataset)
        else:
            test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=args.task_mb_size, num_workers=args.workers, sampler=sampler, shuffle=False)
            test_tasks = args.val_tasks
        checkpoint_file = args.save_dir+'/final_model.pickle'
        meta_learner = pickle.load(open(checkpoint_file, 'rb'))

        losses = np.zeros((test_tasks, 4))
        accuracy = np.zeros((test_tasks, 2))
        meta_params = meta_learner.get_params().clone()

        for i, batch in enumerate(test_dataloader):
            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=device)
            train_targets = train_targets.to(device=device)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=device)
            test_targets = test_targets.to(device=device)
            
            for idx in range(args.task_mb_size):
                fast_learner.set_params(meta_params) # sync weights
                vl_before = fast_learner.get_loss(test_inputs[idx], test_targets[idx], return_numpy=True)
                tl = fast_learner.learn_task(train_inputs[idx], train_targets[idx], num_steps=args.n_steps_val, add_regularization=False)
                fast_learner.inner_opt.zero_grad()
                regu_loss = fast_learner.regularization_loss(meta_params, args.lam)
                regu_loss.backward()
                fast_learner.inner_opt.step()
                vl_after = fast_learner.get_loss(test_inputs[idx], test_targets[idx], return_numpy=True)
                tacc = utils.measure_accuracy(train_inputs[idx], train_targets[idx], fast_learner)
                vacc = utils.measure_accuracy(test_inputs[idx], test_targets[idx], fast_learner)
                losses[i*args.task_mb_size+idx] = np.array([tl[0], vl_before, tl[-1], vl_after])
                accuracy[i*args.task_mb_size+idx][0] = tacc; accuracy[i*args.task_mb_size+idx][1] = vacc
                checkpoint_file = args.save_dir+f'/final_model_task{idx}.pickle'

        accuracy = pd.DataFrame([[np.mean(accuracy, axis=0)[1], 1.96*np.std(accuracy, axis=0)[1]/np.sqrt(test_tasks)]], columns = ['test_acc', 'test_std'])
        accuracy.to_csv(args.save_dir + '/test_accuracy.csv')

if __name__ == '__main__':
    main()