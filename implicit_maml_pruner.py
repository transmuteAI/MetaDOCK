import numpy as np
import pandas as pd
import torch
import utils
import pickle
import pathlib
from absl import app
from absl import flags

from learner_model import Learner, make_conv_network
from utils import seed_all, custom_loss_grad, StaticSampler

from torchmeta.datasets.helpers import cifar_fs, miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader

# ===================
# hyperparameters
# ===================
args = flags.FLAGS

flags.DEFINE_string('dataset', 'cifar_fs', 'One of - cifar_fs, miniimagenet')
flags.DEFINE_string('model', '4conv_1', '4conv_1 / 4conv_2')
flags.DEFINE_string('save_dir', 'exps/pruning', '')
flags.DEFINE_string('load_agent', 'exps/pretraining/final_model.pickle', '')
flags.DEFINE_string('val_tasks_path', 'tasks/num_tasks_600_num_classes_20_ways_5.pkl', '')
flags.DEFINE_string('budget_scheduler_type', None, '')
flags.DEFINE_string('budget_weight_scheduler_type', 'linear', '')
flags.DEFINE_integer('N_way', 5, 'N way classification')
flags.DEFINE_integer('K_shot', 5, 'K shot for training')
flags.DEFINE_integer('n_steps', 16, '')
flags.DEFINE_integer('n_steps_val', 16, '')
flags.DEFINE_integer('meta_steps', 60000, '')
flags.DEFINE_integer('task_mb_size', 4, 'meta batch size')
flags.DEFINE_integer('cg_steps', 5, '')
flags.DEFINE_integer('workers', 8, '')
flags.DEFINE_integer('seed', 10, '')
flags.DEFINE_integer('val_tasks', 600, '')
flags.DEFINE_integer('val_every', 1000, '')
flags.DEFINE_float('inner_lr', 5e-3, '')
flags.DEFINE_float('outer_lr', 5e-5, '')
flags.DEFINE_float('lam', 0.5, '')
flags.DEFINE_float('cg_damping', 1.0, '')
flags.DEFINE_float('budget', 0.5, '')
flags.DEFINE_float('budget_loss_weight', 50, '')
flags.DEFINE_float('l1_weight', 1e-6, '')
flags.DEFINE_bool('use_gpu', True, '')
flags.DEFINE_bool('train', True, '')
flags.DEFINE_bool('test', True, '')
flags.DEFINE_bool('save_zetas', True, '')


def main(argv):
    

    seed_all(args.seed)

    train_dataset = globals()[args.dataset]("data", ways=args.N_way, shots=args.K_shot, test_shots=args.K_shot, meta_train=True, download=True)
    train_dataloader = BatchMetaDataLoader(train_dataset, batch_size=args.task_mb_size, num_workers=args.workers)

    # load pretrained model
    learner_net = make_conv_network(out_dim=args.N_way, task=args.dataset, model_name=args.model, method = 'prune', mode = 'conv')
    fast_net = make_conv_network(out_dim=args.N_way, task=args.dataset, model_name=args.model, method = 'prune', mode = 'conv')
    meta_learner = Learner(model=learner_net, loss_function=torch.nn.CrossEntropyLoss(), inner_alg='bilevel',
                        inner_lr=args.inner_lr, outer_lr=args.outer_lr, GPU=args.use_gpu, budget=args.budget, 
                        w1=args.budget_loss_weight, w2=args.l1_weight, 
                        num_total_iter=args.meta_steps, budget_scheduler_type=args.budget_scheduler_type,
                        budget_weight_scheduler_type=args.budget_weight_scheduler_type)
    fast_learner = Learner(model=fast_net, loss_function=torch.nn.CrossEntropyLoss(), inner_alg='bilevel',
                        inner_lr=args.inner_lr, outer_lr=args.outer_lr, GPU=args.use_gpu, budget=args.budget, 
                        w1=args.budget_loss_weight, w2=args.l1_weight, 
                        num_total_iter=args.meta_steps, budget_scheduler_type=args.budget_scheduler_type,
                        budget_weight_scheduler_type=args.budget_weight_scheduler_type)
    # w0 = torch.load(args.load_agent, map_location=torch.device('cpu'))
    w0 = pickle.load(open(args.load_agent, 'rb')).model.state_dict()
    meta_learner.model.load_state_dict(w0, strict=False)
    fast_learner.model.load_state_dict(w0, strict=False)
    meta_learner.hist['loss'] = np.zeros((args.meta_steps, 4))
    meta_learner.hist['accuracy'] = np.zeros((args.meta_steps, 2))
    meta_learner.hist['val_accuracy'] = np.zeros((args.meta_steps//args.val_every, 2))
    meta_learner.hist['remaining'] = np.zeros((args.meta_steps, 1))
        
    device = 'cuda' if args.use_gpu is True else 'cpu'
    lam = torch.tensor(args.lam)
    lam = lam.to(device)
    best_acc = 0
    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    args.append_flags_into_file(args.save_dir+'/flagfile.txt')


    # ===================
    # Train
    # ===================
    if args.train:
        print("Training model ......")
        for outstep, batch in enumerate(train_dataloader, start=0):
            if outstep==args.meta_steps:
                break
            w_k = meta_learner.get_params()
            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=device)
            train_targets = train_targets.to(device=device)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=device)
            test_targets = test_targets.to(device=device)
            meta_grad = 0.0

            for idx in range(args.task_mb_size):
                fast_learner.curr_iter=meta_learner.curr_iter
                fast_learner.set_params(w_k.clone()) # sync weights

                vl_before = fast_learner.get_loss(test_inputs[idx], test_targets[idx], return_numpy=True, use_budget=True)

                tl = fast_learner.learn_task(train_inputs[idx], train_targets[idx], num_steps=args.n_steps)
                
                # pull back for regularization
                fast_learner.inner_opt.zero_grad()
                regu_loss = fast_learner.regularization_loss(w_k, lam)
                regu_loss.backward()
                fast_learner.inner_opt.step()

                tacc = utils.measure_accuracy(train_inputs[idx], train_targets[idx], fast_learner)
                vacc = utils.measure_accuracy(test_inputs[idx], test_targets[idx], fast_learner)

                valid_loss, valid_grad = custom_loss_grad(test_inputs[idx], test_targets[idx], fast_learner)
                vl_after = utils.to_numpy(valid_loss).ravel()[0]
                flat_grad = torch.cat([g.contiguous().view(-1) for g in valid_grad])
                if args.cg_steps <= 1:
                    task_outer_grad = flat_grad
                else:
                    task_matrix_evaluator = fast_learner.matrix_evaluator(train_inputs[idx], train_targets[idx], lam, args.cg_damping)
                    task_outer_grad = utils.cg_solve(task_matrix_evaluator, flat_grad, args.cg_steps, x_init=None)

                meta_grad += task_outer_grad
                meta_learner.hist['loss'][outstep] += (np.array([tl[0], vl_before, tl[-1], vl_after])/args.task_mb_size)
                meta_learner.hist['accuracy'][outstep] += np.array([tacc, vacc]) / args.task_mb_size
                rem = fast_learner.model.get_remaining().item()
                meta_learner.hist['remaining'][outstep] += np.array([rem]) / args.task_mb_size

            
            meta_learner.outer_step_with_grad(meta_grad/args.task_mb_size, flat_grad=True)

            if (outstep % args.val_every == 0 and outstep > 0) or outstep==args.meta_steps-1:
                loss = pd.DataFrame(meta_learner.hist['loss'], columns = ['train_pre', 'test_pre', 'train_post', 'test_post'])
                accuracy = pd.DataFrame(meta_learner.hist['accuracy'], columns = ['train_acc', 'val_acc'])
                remaining = pd.DataFrame(meta_learner.hist['remaining'], columns = ['remaining'])
                loss.to_csv(args.save_dir + '/loss.csv')
                accuracy.to_csv(args.save_dir + '/accuracy.csv')
                remaining.to_csv(args.save_dir + '/remaining.csv')

            if (outstep % args.val_every == 0) or outstep==args.meta_steps-1:
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

                        vl_before = fast_learner.get_loss(test_inputs[idx], test_targets[idx], return_numpy=True, use_budget=True)
                        tl = fast_learner.learn_task(train_inputs[idx], train_targets[idx], num_steps=args.n_steps, add_regularization=False)
                        fast_learner.inner_opt.zero_grad()
                        regu_loss = fast_learner.regularization_loss(meta_params, args.lam)
                        regu_loss.backward()
                        fast_learner.inner_opt.step()
                        vl_after = fast_learner.get_loss(test_inputs[idx], test_targets[idx], return_numpy=True, use_budget=True)
                        tacc = utils.measure_accuracy(train_inputs[idx], train_targets[idx], fast_learner)
                        vacc = utils.measure_accuracy(test_inputs[idx], test_targets[idx], fast_learner)
                        losses[i*args.task_mb_size+idx] = np.array([tl[0], vl_before, tl[-1], vl_after])
                        accuracy[i*args.task_mb_size+idx][0] = tacc; accuracy[i*args.task_mb_size+idx][1] = vacc

                if np.mean(accuracy, axis=0)[1] >= best_acc:
                    checkpoint_file = args.save_dir+'/final_model.pickle'
                    pickle.dump(meta_learner, open(checkpoint_file, 'wb'), pickle.HIGHEST_PROTOCOL)
                    best_acc = np.mean(accuracy, axis=0)[1]
                meta_learner.hist['val_accuracy'][outstep//args.val_every][0] = np.mean(accuracy, axis=0)[1]
                meta_learner.hist['val_accuracy'][outstep//args.val_every][1] = 1.96*np.std(accuracy, axis=0)[1]/np.sqrt(args.val_tasks)
                val_accuracy = pd.DataFrame(meta_learner.hist['val_accuracy'], columns = ['val_acc', 'val_std'])
                val_accuracy.to_csv(args.save_dir + '/val_accuracy.csv')

    test_dataset = globals()[args.dataset]("data", ways=args.N_way, shots=args.K_shot, test_shots=args.K_shot, meta_test=True, download=True)
    
    checkpoint_file = args.save_dir+'/final_model.pickle'
    meta_learner = pickle.load(open(checkpoint_file, 'rb'))
    fast_learner = pickle.load(open(checkpoint_file, 'rb'))
    
    test_tasks = len(test_dataset)
    if args.save_zetas:
        print("Saving Zetas .....")
        sampler = StaticSampler(args.val_tasks_path)
        test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=args.task_mb_size, num_workers=args.workers, sampler=sampler, shuffle=False)
        test_tasks = args.val_tasks
        
        losses = np.zeros((test_tasks, 4))
        accuracy = np.zeros((test_tasks, 2))
        zeta_t_list = []
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
                vl_before = fast_learner.get_loss(test_inputs[idx], test_targets[idx], return_numpy=True, use_budget=True)
                tl = fast_learner.learn_task(train_inputs[idx], train_targets[idx], num_steps=args.n_steps, add_regularization=False)
                fast_learner.inner_opt.zero_grad()
                regu_loss = fast_learner.regularization_loss(meta_params, args.lam)
                regu_loss.backward()
                fast_learner.inner_opt.step()
                vl_after = fast_learner.get_loss(test_inputs[idx], test_targets[idx], return_numpy=True, use_budget=True)
                tacc = utils.measure_accuracy(train_inputs[idx], train_targets[idx], fast_learner)
                vacc = utils.measure_accuracy(test_inputs[idx], test_targets[idx], fast_learner)
                losses[i*args.task_mb_size+idx] = np.array([tl[0], vl_before, tl[-1], vl_after])
                accuracy[i*args.task_mb_size+idx][0] = tacc; accuracy[i*args.task_mb_size+idx][1] = vacc
                if args.save_zetas:
                    zeta_t_list.append((np.array(fast_learner.model.give_zetas())))
        accuracy = pd.DataFrame([[np.mean(accuracy, axis=0)[1], 1.96 *np.std(accuracy, axis=0)[1]/np.sqrt(test_tasks)]], columns = ['test_acc', 'test_std'])
        accuracy.to_csv(args.save_dir + '/final_val_accuracy.csv')
        pickle.dump(zeta_t_list, open(args.save_dir+'/final_val_zetas.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

    if args.test:
        print("Testing model ......")
        test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=args.task_mb_size, num_workers=args.workers, shuffle=False)
        test_tasks = len(test_dataset)
        if args.train:
            checkpoint_file = args.save_dir+'/final_model.pickle'
        else:
            checkpoint_file = args.load_agent
        test_tasks = len(test_dataset)
        meta_learner = pickle.load(open(checkpoint_file, 'rb'))
        fast_learner = pickle.load(open(checkpoint_file, 'rb'))

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
                vl_before = fast_learner.get_loss(test_inputs[idx], test_targets[idx], return_numpy=True, use_budget=True)
                tl = fast_learner.learn_task(train_inputs[idx], train_targets[idx], num_steps=args.n_steps, add_regularization=False)
                fast_learner.inner_opt.zero_grad()
                regu_loss = fast_learner.regularization_loss(meta_params, args.lam)
                regu_loss.backward()
                fast_learner.inner_opt.step()
                vl_after = fast_learner.get_loss(test_inputs[idx], test_targets[idx], return_numpy=True, use_budget=True)
                tacc = utils.measure_accuracy(train_inputs[idx], train_targets[idx], fast_learner)
                vacc = utils.measure_accuracy(test_inputs[idx], test_targets[idx], fast_learner)
                losses[i*args.task_mb_size+idx] = np.array([tl[0], vl_before, tl[-1], vl_after])
                accuracy[i*args.task_mb_size+idx][0] = tacc; accuracy[i*args.task_mb_size+idx][1] = vacc
        accuracy = pd.DataFrame([[np.mean(accuracy, axis=0)[1], 1.96 *np.std(accuracy, axis=0)[1]/np.sqrt(test_tasks)]], columns = ['test_acc', 'test_std'])
        accuracy.to_csv(args.save_dir + '/test_accuracy.csv')

if __name__ == '__main__':
    app.run(main)