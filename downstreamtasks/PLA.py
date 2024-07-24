from Utils import *

# Define the Gaussian Process model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, X_train, y_train, likelihood):
        super(ExactGPModel, self).__init__(X_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train(model, X_train, y_train, likelihood, data_folder, modal):
    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Train the model
    training_iter = 100
    for i in range(training_iter):
        model.train()
        likelihood.train()
        optimizer.zero_grad()
        # Compute the output from the model
        output = model(X_train)
        # Calculate the loss and backpropagate gradients
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
    
    # Save the trained model state
    torch.save(model.state_dict(), f'{data_folder}/{modal}_model_state.pth')

def test(model, X_test, y_test, likelihood, data_folder, modal):
    # Load the trained model state
    state_dict = torch.load(f'{data_folder}/{modal}_model_state.pth')
    model.load_state_dict(state_dict)

    # Evaluate the model on the test set
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(X_test))
        mse = gpytorch.metrics.mean_squared_error(observed_pred, y_test, squared=True).item()
        mae = gpytorch.metrics.mean_absolute_error(observed_pred, y_test).item()
        ci = get_cindex(observed_pred.loc, y_test)
        rm2 = get_rm2(observed_pred.loc, y_test)
        pearsonr = get_pearson(observed_pred.loc, y_test)
        spearmanr = get_spearman(observed_pred.loc, y_test)

        lower, upper = observed_pred.confidence_region()

        # Print evaluation metrics
        print("Mean Squared Error:", mse)
        print("Mean Absolute Error:", mae)
        print("Root Mean Square Error: ", sqrt(mse))
        print("Pearson Correlation:", pearsonr)
        print("Spearman Correlation:", spearmanr)
        print("C-Index:", ci)
        print("Rm^2:", rm2)
        print("Mean of Lower Confidence Bounds:", lower.mean())
        print("Mean of Upper Confidence Bounds:", upper.mean())

def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description='Gaussian Process model for different modalities and datasets.')
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help='Mode: train or test')
    parser.add_argument('--modal', choices=['sequence', 'graph', 'point_cloud', 'multimodal'], required=True, help='Modality to use')
    parser.add_argument('--dataset', choices=['KIBA', 'DAVIS', 'PDBbind'], required=True, help='Dataset to use')

    # Parse command-line arguments
    args = parser.parse_args()
    data_folder = f'./downstreamtasks/data/{args.dataset}'

    # Read the label CSV file
    df = pd.read_csv(f'{data_folder}label.csv')
    print("Number of samples:", len(df))

    # Load feature data from the selected modality
    with open(f'{data_folder}/{args.modal}.pkl', 'rb') as f:
        X = np.array(pickle.load(f))

    # Load labels
    y = df['label'].to_numpy()

    # Load test and train IDs from the fold setting
    with open(f'{data_folder}/folds/test_fold_setting.txt', 'r') as f:
        test_ids_str = f.read()
        test_ids = ast.literal_eval(test_ids_str)
        train_ids = np.setdiff1d(np.arange(X.shape[0]), test_ids)

    # Print the sizes of the training and test sets
    print(f'Size of Training Set: {len(train_ids)} samples')
    print(f'Size of Test Set: {len(test_ids)} samples')

    # Split the data into training and test sets
    X_train = X[train_ids]
    y_train = y[train_ids]

    X_test = X[test_ids]
    y_test = y[test_ids]

    # Convert data to PyTorch tensors
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

    # Create a Gaussian likelihood for the model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # Initialize the Gaussian Process model
    model = ExactGPModel(X_train, y_train, likelihood)

    # Perform either training or testing based on the mode argument
    if args.mode == 'train':
        train(model, X_train, y_train, likelihood, data_folder, args.modal)
    elif args.mode == 'test':
        test(model, X_test, y_test, likelihood, data_folder, args.modal)

if __name__ == '__main__':
    main()
