"""
main program for running internal pre-training and CLNN

some functions are modified from
https://github.com/thuiar/DeepAligned-Clustering/blob/main/DeepAligned.py
"""

from model import CLBert
from init_parameter import init_model
from dataloader import Data
from mtp import InternalPretrainModelManager
from utils.tools import *
from utils.memory import MemoryBank, fill_memory_bank
from utils.neighbor_dataset import NeighborsDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CLNNModelManager:
    """
    The implementation of Contrastive Learning with Nearest Neighbors
    """
    def __init__(self, args, data, pretrained_model=None):
        set_seed(args.seed)
        n_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = data.num_labels
        self.model = CLBert(args.bert_model, device=self.device)

        if n_gpu > 1:
            self.model = nn.DataParallel(self.model)
        
        if not args.disable_pretrain:
            self.pretrained_model = pretrained_model
            self.load_pretrained_model()
        
        self.num_train_optimization_steps = int(len(data.train_semi_dataset) / args.train_batch_size) * args.num_train_epochs
        
        self.optimizer, self.scheduler = self.get_optimizer(args)
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.generator = view_generator(self.tokenizer, args.rtr_prob, args.seed)

    def get_neighbor_dataset(self, args, data, indices):
        """convert indices to dataset"""
        dataset = NeighborsDataset(data.train_semi_dataset, indices)
        self.train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    
    def get_neighbor_inds(self, args, data):
        """get indices of neighbors"""
        memory_bank = MemoryBank(len(data.train_semi_dataset), args.feat_dim, len(data.all_label_list), 0.1)
        fill_memory_bank(data.train_semi_dataloader, self.model, memory_bank)
        indices = memory_bank.mine_nearest_neighbors(args.topk, calculate_accuracy=False)
        return indices
    
    def get_adjacency(self, args, inds, neighbors, targets):
        """get adjacency matrix"""
        adj = torch.zeros(inds.shape[0], inds.shape[0])
        for b1, n in enumerate(neighbors):
            adj[b1][b1] = 1
            for b2, j in enumerate(inds):
                if j in n:
                    adj[b1][b2] = 1 # if in neighbors
                if (targets[b1] == targets[b2]) and (targets[b1]>0) and (targets[b2]>0):
                    adj[b1][b2] = 1 # if same labels
                    # this is useful only when both have labels
        return adj

    def evaluation(self, args, data, save_results=True, plot_cm=True):
        """final clustering evaluation on test set"""
        # get features
        feats_test, labels = self.get_features_labels(data.test_dataloader, self.model, args)
        feats_test = feats_test.cpu().numpy()

        # k-means clustering
        km = KMeans(n_clusters = self.num_labels).fit(feats_test)
        
        y_pred = km.labels_
        y_true = labels.cpu().numpy()

        results = clustering_score(y_true, y_pred)
        print('results',results)
        
        # confusion matrix
        if plot_cm:
            ind, _ = hungray_aligment(y_true, y_pred)
            map_ = {i[0]:i[1] for i in ind}
            y_pred = np.array([map_[idx] for idx in y_pred])

            cm = confusion_matrix(y_true,y_pred)   
            print('confusion matrix',cm)
            self.test_results = results
        
        # save results
        if save_results:
            self.save_results(args)

    def train(self, args, data):

        if isinstance(self.model, nn.DataParallel):
            criterion = self.model.module.loss_cl
        else:
            criterion = self.model.loss_cl
        
        # load neighbors for the first epoch
        indices = self.get_neighbor_inds(args, data)
        self.get_neighbor_dataset(args, data, indices)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for batch in tqdm(self.train_dataloader, desc="Iteration"):
                # 1. load data
                anchor = tuple(t.to(self.device) for t in batch["anchor"]) # anchor data
                neighbor = tuple(t.to(self.device) for t in batch["neighbor"]) # neighbor data
                pos_neighbors = batch["possible_neighbors"] # all possible neighbor inds for anchor
                data_inds = batch["index"] # neighbor data ind

                # 2. get adjacency matrix
                adjacency = self.get_adjacency(args, data_inds, pos_neighbors, batch["target"]) # (bz,bz)

                # 3. get augmentations
                if args.view_strategy == "rtr":
                    X_an = {"input_ids":self.generator.random_token_replace(anchor[0].cpu()).to(self.device), "attention_mask":anchor[1], "token_type_ids":anchor[2]}
                    X_ng = {"input_ids":self.generator.random_token_replace(neighbor[0].cpu()).to(self.device), "attention_mask":neighbor[1], "token_type_ids":neighbor[2]}
                elif args.view_strategy == "shuffle":
                    X_an = {"input_ids":self.generator.shuffle_tokens(anchor[0].cpu()).to(self.device), "attention_mask":anchor[1], "token_type_ids":anchor[2]}
                    X_ng = {"input_ids":self.generator.shuffle_tokens(neighbor[0].cpu()).to(self.device), "attention_mask":neighbor[1], "token_type_ids":neighbor[2]}
                elif args.view_strategy == "none":
                    X_an = {"input_ids":anchor[0], "attention_mask":anchor[1], "token_type_ids":anchor[2]}
                    X_ng = {"input_ids":neighbor[0], "attention_mask":neighbor[1], "token_type_ids":neighbor[2]}
                else:
                    raise NotImplementedError(f"View strategy {args.view_strategy} not implemented!")
                
                # 4. compute loss and update parameters
                with torch.set_grad_enabled(True):
                    f_pos = torch.stack([self.model(X_an)["features"], self.model(X_ng)["features"]], dim=1)
                    loss = criterion(f_pos, mask=adjacency, temperature=args.temp)
                    tr_loss += loss.item()
                    
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    nb_tr_examples += anchor[0].size(0)
                    nb_tr_steps += 1
            
            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)
                        
            # update neighbors every several epochs
            if ((epoch + 1) % args.update_per_epoch) == 0:
                indices = self.get_neighbor_inds(args, data)
                self.get_neighbor_dataset(args, data, indices)

    def get_optimizer(self, args):
        num_warmup_steps = int(args.warmup_proportion*self.num_train_optimization_steps)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=self.num_train_optimization_steps)
        return optimizer, scheduler
    
    def load_pretrained_model(self):
        """load the backbone of pretrained model"""
        if isinstance(self.pretrained_model, nn.DataParallel):
            pretrained_dict = self.pretrained_model.module.backbone.state_dict()
        else:
            pretrained_dict = self.pretrained_model.backbone.state_dict()
        if isinstance(self.model, nn.DataParallel):
            self.model.module.backbone.load_state_dict(pretrained_dict, strict=False)
        else:
            self.model.backbone.load_state_dict(pretrained_dict, strict=False)

    def get_features_labels(self, dataloader, model, args):
        model.eval()
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.no_grad():
                feature = model(X, output_hidden_states=True)["hidden_states"]

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels
            
    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.topk, args.view_strategy, args.seed]
        names = ['dataset', 'method', 'known_cls_ratio', 'labeled_ratio', 'topk', 'view_strategy', 'seed']
        vars_dict = {k:v for k,v in zip(names, var)}
        results = dict(self.test_results,**vars_dict)
        keys = list(results.keys())
        values = list(results.values())
        
        file_name = 'results.csv'
        results_path = os.path.join(args.save_results_path, file_name)
        
        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori,columns = keys)
            df1.to_csv(results_path,index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results,index=[1])
            df1 = df1.append(new,ignore_index=True)
            df1.to_csv(results_path,index=False)
        data_diagram = pd.read_csv(results_path)
        
        print('test_results', data_diagram)

if __name__ == '__main__':

    print('Data and Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()
    print(args)

    if args.known_cls_ratio == 0:
        args.disable_pretrain = True # disable internal pretrain
    else:
        args.disable_pretrain = False

    if not args.disable_pretrain:
        data = Data(args)
        print('Pre-training begin...')
        manager_p = InternalPretrainModelManager(args, data)
        manager_p.train(args, data)
        print('Pre-training finished!')
        manager = CLNNModelManager(args, data, manager_p.model) # pass the model to clnn
    else:
        data = Data(args)
        manager = CLNNModelManager(args, data)
    
    if args.report_pretrain:
        method = args.method
        args.method = 'pretrain'
        manager.evaluation(args, data) # evaluate when report performance on pretrain

        args.method = method

    print('Training begin...')
    manager.train(args,data)
    print('Training finished!')

    print('Evaluation begin...')
    manager.evaluation(args, data)
    print('Evaluation finished!')

    print('Saving Model ...')
    if args.save_model_path:
        manager.model.save_backbone(args.save_model_path)
    print("Finished!")