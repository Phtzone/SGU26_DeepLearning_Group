"""Keep best model selection separate from significant-improvement patience."""
class SelectionState:
    def __init__(self, patience, min_delta):
        self.patience, self.min_delta = patience, min_delta
        self.best_score, self.best_loss, self.anchor = -1.0, float('inf'), -1.0
        self.best_epoch, self.bad_epochs = 0, 0

    def update(self, score, loss, epoch):
        better = score > self.best_score + 1e-12 or (abs(score - self.best_score) <= 1e-12 and loss < self.best_loss)
        if better:
            self.best_score, self.best_loss, self.best_epoch = score, loss, epoch
        if score > self.anchor + self.min_delta:
            self.anchor, self.bad_epochs = score, 0
        else:
            self.bad_epochs += 1
        return better, self.bad_epochs >= self.patience
