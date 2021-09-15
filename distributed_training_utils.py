import torch
import os
import torch.optim as optim
import compression_utils as comp
from metric_utils.metric import Metric
from loss_utils.kl_divergence import kl_divergence
from loss_utils.focol_tversky_loss import TverskyCrossEntropyDiceWeightedLoss
import torch.nn.functional as F
from torch import autograd

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()


def copy_decoder(target, source):
    for name in target:
        if "down_convs" not in name:
            target[name].data = source[name].data.clone()


def add(target, source):
    for name in target:
        target[name].data += source[name].data.clone()


def scale(target, scaling):
    for name in target:
        target[name].data = scaling * target[name].data.clone()


def subtract(target, source):
    for name in target:
        target[name].data -= source[name].data.clone()


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


def add_subtract(target, minuend, subtrahend):
    for name in target:
        target[name].data = target[name].data + minuend[name].data.clone() - subtrahend[name].data.clone()


def average(target, sources):
    for name in target:
        target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()
        

def average_decoder(target, sources):
    for name in target:
        if "down_convs" not in name:
            target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()


def weighted_average(target, sources, weights):
    for name in target:
        summ = torch.sum(weights)
        n = len(sources)
        modify = [weight / summ * n for weight in weights]
        target[name].data = torch.mean(torch.stack([m * source[name].data for source, m in zip(sources, modify)]),
                                       dim=0).clone()

def weighted_average_decoder(target, sources, weights):
    for name in target:
        if "down_convs" not in name:
            summ = torch.sum(weights)
            n = len(sources)
            modify = [weight / summ * n for weight in weights]
            target[name].data = torch.mean(torch.stack([m * source[name].data for source, m in zip(sources, modify)]),
                                           dim=0).clone()


def weighted_weights(target, source, alpha=0.25):
    for name in target:
        target[name].data = alpha * target[name].data.clone() + (1 - alpha) * source[name].data.clone()


def majority_vote(target, sources, lr):
    for name in target:
        mask = torch.stack([source[name].data.sign() for source in sources]).sum(dim=0).sign()
        target[name].data = (lr * mask).clone()


def compress(target, source, compress_fun):
    """compress_fun : a function f : tensor (shape) -> tensor (shape)"""
    for name in target:
        target[name].data = compress_fun(source[name].data.clone())


class DistributedTrainingDevice(object):
    def __init__(self, train_loader, val_loader, model, hyperparameters, experiment):
        self.hp = hyperparameters
        self.xp = experiment
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.loss_fn = TverskyCrossEntropyDiceWeightedLoss(num_class=2, device=device)


class Client(DistributedTrainingDevice):

    def __init__(self, train_loader, val_loader, model, hyperparameters, experiment, name, num_id):
        super().__init__(train_loader, val_loader, model, hyperparameters, experiment)

        self.name = name
        self.id = num_id

        # Parameters
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.W_old = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.dW_compressed = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.A = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}

        self.n_params = sum([T.numel() for T in self.W.values()])
        self.bits_sent = []

        # Optimizer (specified in self.hp, initialized using the suitable parameters from self.hp)
        optimizer_object = getattr(optim, self.hp['optimizer'])
        optimizer_parameters = {k: v for k, v in self.hp.items() if k in optimizer_object.__init__.__code__.co_varnames}

        self.optimizer = optimizer_object(self.model.parameters(), **optimizer_parameters)

        # Learning Rate Schedule
        self.scheduler = getattr(optim.lr_scheduler, self.hp['lr_decay'][0])(self.optimizer, **self.hp['lr_decay'][1])

        # State
        self.epoch = 0
        self.train_loss = 0.0
        self.metric = Metric(num_class=2)
        self.global_prototype = None
        self.local_prototype = None

    def synchronize_with_server(self, server):

        # W_client = W_server
        copy(target=self.W, source=server.W)

        # synchronize prototype with the server
        if server.global_prototype is not None:
            self.global_prototype = server.global_prototype.data.clone()

    def getFeatures(self, fm, mask):
        """
        Extract foreground features via masked average pooling

        Args:
            fm: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fm = F.interpolate(fm, size=mask.shape[-2:], mode='bilinear', align_corners=True)
        masked_fm = torch.sum(fm * mask[None, ...], dim=(2, 3)) / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)
        return masked_fm

    def getPrototypes(self, features):
        """
        Average the features to obtain the prototype
        :param features: list of foreground features for the building
        :return: prototype for the building class
        """
        local_prototype = sum(features) / len(features)
        return local_prototype

    def calDist(self, fm, prototype, lamda=1):
        """
        :param fm: input features, shape 1 * C
        :param prototype: prototype of the global model, shape 1 * C
        :return: Cosine Distance between two vectors
        """
        if fm is None or prototype is None:
            return 0
        dist = F.cosine_similarity(fm, prototype) * lamda
        return dist

    def train_cnn(self, iterations):

        running_loss = 0.0

        for i in range(iterations):

            self.epoch += 1

            for j, (x, y) in enumerate(self.train_loader):

                # x shape: N * C * H * W, y shape N * H * W
                x, y = x.to(device), y.to(device).long()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                y_, fm = self.model(x)

                # calculate local features
                features = []
                for epi in range(len(x)):
                    feature = self.getFeatures(fm[[epi]], y[[epi]])
                    features.append(feature)

                # calculate local prototypes
                pre_prototype = self.getPrototypes(features)

                # assign the local_prototype from last iteration as the prototype for this round of communication
                if i == iterations - 1:
                    self.local_prototype = pre_prototype

                # calculate the distance
                prototype_loss = self.calDist(pre_prototype, self.global_prototype)

                loss = self.loss_fn(y_, y) + prototype_loss
                loss.backward(retain_graph=True)
                self.optimizer.step()

                running_loss += loss.item()

            # Adapt lr according to schedule
            if isinstance(self.scheduler, optim.lr_scheduler.LambdaLR):
                self.scheduler.step()
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau) and 'loss_test' in self.xp.results:
                self.scheduler.step(self.xp.results['loss_test'][-1])

        running_loss /= len(self.train_loader)
        return running_loss / iterations

    def compute_weight_update(self, iterations=1):

        # Training mode
        self.model.train()

        # W_old = W
        copy(target=self.W_old, source=self.W)

        # W = SGD(W, D)
        self.train_loss = self.train_cnn(iterations)
        print("Training loss at epoch {} of Client {} is {:3f}".format(self.epoch, self.name, self.train_loss))

        # dW = W - W_old
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)

    def compress_weight_update_up(self, compression=None, accumulate=False, count_bits=False):

        if accumulate and compression[0] != "none":
            # compression with error accumulation
            add(target=self.A, source=self.dW)
            compress(target=self.dW_compressed, source=self.A, compress_fun=comp.compression_function(*compression))
            subtract(target=self.A, source=self.dW_compressed)

        else:
            # compression without error accumulation
            compress(target=self.dW_compressed, source=self.dW, compress_fun=comp.compression_function(*compression))

        if count_bits:
            # Compute the update size
            self.bits_sent += [comp.get_update_size(self.dW_compressed, compression)]

    def update_fisher_params(self, num_batch):
        log_likelihoods = []
        for i, (x, y) in enumerate(self.train_loader):
            # define the number of samples
            if i >= num_batch:
                break
            x, y = x.to(device), y.to(device).long()
            output = F.log_softmax(self.model(x), dim=1)
            log_likelihoods.append(output[:, y])
        log_likelihood = torch.cat(log_likelihoods).mean()
        grad_log_likelihood = autograd.grad(log_likelihood, self.model.parameters())
        _buff_param_names = [param[0].replace('.', '_') for param in self.model.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_likelihood):
            self.model.register_buffer(_buff_param_name + '_estimated_fisher', param.data.clone() ** 2)

    def update_mean_params(self):
        for param_name, params in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '_')
            self.model.register_buffer(_buff_param_name + '_estimated_mean', params)

    def evaluate(self, writer):
        if self.val_loader is None:
            return

        self.model.eval()
        self.metric.reset()

        for i, (x, y) in enumerate(self.val_loader):
            x, y = x.to(device), y.to(device).long()
            y_, fm = self.model(x)
            _, predicted = torch.max(y_, 1)

            self.metric.add_pixel_accuracy(predicted, y)
            self.metric.add_confusion_matrix(predicted, y)

        iou, mean_iou = self.metric.iou_value()
        accuracy = self.metric.accuracy_value()

        writer.add_scalar('Accuracy/' + self.name, accuracy, self.epoch)
        writer.add_scalar('Background IoU/' + self.name, iou[0], self.epoch)
        writer.add_scalar('Building IoU/' + self.name, iou[1], self.epoch)


class Server(DistributedTrainingDevice):

    def __init__(self, train_loader, val_loader, model, hyperparameters, experiment, stats):
        super().__init__(train_loader, val_loader, model, hyperparameters, experiment)

        # Parameters
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.dW_compressed = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}

        self.A = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}

        self.n_params = sum([T.numel() for T in self.W.values()])
        self.bits_sent = []

        self.client_sizes = torch.Tensor(stats["split"]).cuda()
        self.metric = Metric(num_class=2)
        self.global_prototype = None

    def average_prototypes(self, clients):
        """Get global Prototypes"""
        self.global_prototype = torch.mean(torch.stack([client.local_prototype for client in clients]), dim=0).clone()

    def aggregate_weight_updates(self, clients, aggregation="mean"):

        # dW = aggregate(dW_i, i=1,..,n)
        if aggregation == "mean":
            average(target=self.dW, sources=[client.dW_compressed for client in clients])

        elif aggregation == "weighted_mean":
            weighted_average(target=self.dW, sources=[client.dW_compressed for client in clients],
                             weights=torch.stack([self.client_sizes[client.id] for client in clients]))

        elif aggregation == "majority":
            majority_vote(target=self.dW, sources=[client.dW_compressed for client in clients], lr=self.hp["lr"])

    def compress_weight_update_down(self, compression=None, accumulate=False, count_bits=False):
        if accumulate and compression[0] != "none":
            # compression with error accumulation
            add(target=self.A, source=self.dW)
            compress(target=self.dW_compressed, source=self.A, compress_fun=comp.compression_function(*compression))
            subtract(target=self.A, source=self.dW_compressed)

        else:
            # compression without error accumulation
            compress(target=self.dW_compressed, source=self.dW, compress_fun=comp.compression_function(*compression))

        add(target=self.W, source=self.dW_compressed)

        if count_bits:
            # Compute the update size
            self.bits_sent += [comp.get_update_size(self.dW_compressed, compression)]

    def ensemble_models(self, clients, aggregation="mean", iterations=3, distill_loader=None):
        self.aggregate_weight_updates(clients, aggregation)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=5e-7, momentum=0.9)
        loss_func = kl_divergence()
        num_clients = len(clients)
        print("Start updating in the server side.")
        self.model.train()

        for iter in range(iterations):
            for i, img in enumerate(distill_loader):
                ensemble_result = torch.zeros([1, 3, 512, 512]).to(device)
                img = img.to(device)
                server_preds = self.model(img)
                for client in clients:
                    clients_preds = client.model(img)
                    ensemble_result += clients_preds
                ensemble_result /= num_clients
                loss = loss_func(server_preds, ensemble_result / 4.0)
                loss.backward()
                optimizer.step()

    def evaluate(self, loader=None, iter=0, cities=None):
        """Evaluates local model stored in self.W on local dataset or other 'loader if specified
     and returns a dict containing all evaluation metrics"""

        self.model.eval()
        results_dict = {}

        if not loader:
            loader = self.val_loader

        for city in cities:
            self.metric.reset()
            for i, (x, y) in enumerate(loader[city]):
                x, y = x.to(device), y.to(device).long()
                y_ = self.model(x)
                _, predicted = torch.max(y_, 1)

                self.metric.add_pixel_accuracy(predicted, y)
                self.metric.add_confusion_matrix(predicted, y)

            iou, mean_iou = self.metric.iou_value()
            accuracy = self.metric.accuracy_value()
            results_dict[city] = {'accuracy': accuracy, 'Background IoU': iou[0], 'Building IoU': iou[1]}

        checkpoint_dir = os.path.join("result", "distributed")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_name = os.path.join(checkpoint_dir, 'model_meta_learning_{}.pth.tar'.format(iter))
        torch.save(self.model.state_dict(), checkpoint_name)
        print('\r[INFO] Checkpoint has been saved: %s\n' % checkpoint_name)

        return results_dict
