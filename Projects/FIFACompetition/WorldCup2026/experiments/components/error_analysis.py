import numpy as np
import numpy.typing as npt
from .evaluations import score_to_index
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math

class ErrorAnalysis:
    def get_tp_fp_fn_of_class(
        match_result: str,
        features: npt.NDArray,
        labels: npt.NDArray,
        predictions: npt.NDArray
    ): 
        """ 
        Function to get true positives, false positives and false negatives based on match result given and model predictions
        
        Args:
            match_result (scalar): Match result that we want to analyze
            features (ndarray): array with m training features that model used to get predictions
            labels (ndarray): a (121, m) array with m actual labels of the provided features
            predictions (ndarray): a (121, m) array with m predicted labels
            
        Returns:
            TP (ndarray): array with x features that were predicted correctly
            FP (ndarray): array with y features that were falsely predicted to have match result
            FN (ndarray): array with z features that were falsely predicted to not have match result
            all_class_indices (ndarray): a (y,) array with actual class labels for FP
            TPFPFN (ndarray): array with x + y + z features that is combinations of TP, FP and FN
        """
        m = features.shape[1]
        match_result_index = score_to_index(match_result)
        
        TP_indexes = []
        FP_indexes = []
        FN_indexes = []
        all_class_indices = []
        TPFPFN_indexes = []
        
        for i in range(m):
            prediction = predictions[: ,i]
            predicted_index = np.argmax(prediction)
        
            actual = labels[:, i]
            actual_indx = np.argmax(actual)
            
            # Get TPs i.e. prediction = label and label == match
            if (predicted_index == actual_indx and actual_indx == match_result_index):
                TP_indexes.append(i)
                TPFPFN_indexes.append(i)
                all_class_indices.append(actual_indx)
            # Get FN i.e prediction != label and label == prediction
            elif (predicted_index != actual_indx and actual_indx == match_result_index):
                FN_indexes.append(i)
                TPFPFN_indexes.append(i)
                all_class_indices.append(actual_indx)
            # Get FP i.e prediction = match but label != match
            elif (predicted_index == match_result_index and actual_indx != match_result_index):
                FP_indexes.append(i)
                TPFPFN_indexes.append(i)
                all_class_indices.append(actual_indx)
                
        TP = features[:, TP_indexes]
        FP = features[:, FP_indexes]
        FN = features[: , FN_indexes]
        TPFPFN = features[:, TPFPFN_indexes]
        all_class_indices = np.array(all_class_indices)
        
        return (TP, FP, FN, all_class_indices, TPFPFN)
    
    def plot_comparison_fp_fn_tp_with_actual(
        TP: npt.NDArray,
        FP: npt.NDArray,
        FN: npt.NDArray,
        all_items: npt.NDArray,
        all_class_indices: npt.NDArray
    ):
        """ 
        Plot TP, FP and FN on a scatter plot to see if they're related. PCA compression will be used to reduce space to 2 dimensions
        
        Args:
            TP (ndarray): array with x TP values - green
            FP (ndarray): array with y FP values - red
            FN (ndarray): array with z FN values - gray
        """
        # PCA expects last value in shape to be number of features, that is not the case in my dataset so I need to transpose
        TP = TP.T
        FP = FP.T
        FN = FN.T
        all_items = all_items.T
        
        TP_has_content = TP.shape[0] > 0
        FP_has_content = FP.shape[0] > 0
        FN_has_content = FN.shape[0] > 0
        
        pca = PCA(n_components=2)
        if all_items.shape[0] > 2:
            pca.fit(all_items)
        else:
            raise Exception("Not enough data to compress")
        
        print(np.sum(pca.explained_variance_ratio_))
        
        TP_compressed = pca.transform(TP) if TP_has_content else []
        FP_compressed = pca.transform(FP) if FP_has_content else []
        FN_compressed = pca.transform(FN) if FN_has_content else []
        All_compressed = pca.transform(all_items)
        
        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(20,8))
        
        # Left subplot with FP, TP, FN
        left_plot = axs[1]
        if TP_has_content: left_plot.scatter(TP_compressed[:, 0], TP_compressed[:, 1], color='green', label='TP', alpha=0.7, edgecolors='none')
        if FP_has_content: left_plot.scatter(FP_compressed[:, 0], FP_compressed[:, 1], color='red', label='FP', alpha=0.7, edgecolors='none')
        if FN_has_content: left_plot.scatter(FN_compressed[:, 0], FN_compressed[:, 1], color='grey', label='FN', alpha=0.7, edgecolors='none')
        
        left_plot.set_title('PCA Projection of TP, FP, FN Comparison', fontsize=14, fontweight='bold')
        left_plot.set_xlabel('Principal Component 1', fontsize=12)
        left_plot.set_ylabel('Principal Component 2', fontsize=12)
        left_plot.grid(True, linestyle='--', alpha=0.5)
        left_plot.legend(loc='best', fontsize=11)  # Automatically maps colors to labels
        
        # Right subplot with actual values
        right_plot = axs[0]
        label_lookup = [f"{math.floor(i / 11)} -{i%11}" for i in range(121)]
        scatter = right_plot.scatter(
            All_compressed[:, 0], 
            All_compressed[:, 1], 
            c=all_class_indices, 
            cmap='turbo', 
            alpha=0.8, 
            edgecolors='none',
            s=30
        )
        cbar = fig.colorbar(scatter, ticks=np.linspace(0, 120, 11))
        cbar.set_label('Match Result', fontsize=12)
        
        tick_positions = np.linspace(0, 120, 6, dtype=int)
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels([label_lookup[idx] for idx in tick_positions])
        
        right_plot.set_title('Classes of FP', fontsize=14, fontweight='bold')
        right_plot.set_xlabel('Principal Component 1', fontsize=11)
        right_plot.set_ylabel('Principal Component 2', fontsize=11)
        right_plot.grid(True, linestyle='--', alpha=0.3)
            
        # 3. Clean up spacing and render
        plt.tight_layout()
        plt.show()
