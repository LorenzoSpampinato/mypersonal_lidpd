from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='Standardize Parkinson data')
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument("--data_path", type=str,
                        help='Absolute path to the folder containing PSG data',
                        #default=r'.preprocessed_post_ICA_fif')
                        default=r'./Preprocessing_post_ICA_HP_0.5Hz')
                        #default=r'./Dataset_bin')
                        #default= "D:\TESI\lid-data-samples\lid-data-samples\Dataset")
    parser.add_argument("--label_path", type=str,
                        help='Absolute path to the folder containing the labels',
                        default=r'./Annotations')
                        #default= "D:\TESI\lid-data-samples\lid-data-samples\Labels")
    parser.add_argument("--save_path", type=str,
                        help='Absolute path where to save the pre-processed PSG data',
                        default=r'./Results')
                        #default= "D:\TESI\lid-data-samples\lid-data-samples\Results")
    parser.add_argument("--info_path", type=str,
                        help='Absolute path where to save the logfiles',
                        default=r'./Results')
                        #default= "D:\TESI\lid-data-samples\lid-data-samples\Results")
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument("--run_hypnogram", type=bool,
                        help='Flag to run hypnogram definition',
                        default=False)
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument("--run_preprocess", type=bool,
                        help='Flag to run pre-processing',
                        default=False)
    parser.add_argument("--run_bad_interpolation", type=bool,
                        help='Flag to run bad channels interpolation',
                        default=False)
    parser.add_argument("--run_features", type=bool,
                        help='Flag to run feature extraction',
                        default=True)
    parser.add_argument("--only_class", type=str,
                        help='Disease stage only for which features are extracted (i.e., DNV, ADV, DYS, CTL)',
                        default='DYS')
    parser.add_argument("--only_patient", type=str,
                        help='Patient name only for which features are extracted (i.e., PD002, PD003, ...)',
                        default='PD044,PD045')
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument("--run_aggregation", type=bool,
                        help='Flag to run data aggregation for dataframe definition (for each subject separately)',
                        default=False)
    parser.add_argument("--aggregate_labels", type=bool,
                        help='Flag to whether consider labels during dataframe definition',
                        default=False)
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument("--run_nan_check", type=bool,
                        help='Flag to run check on NaN values among extracted features',
                        default=False)
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument("--run_dataset_split", type=bool,
                        help='Flag to run Training, Validation, and Test Set definition',
                        default=False)
    parser.add_argument("--training_percentage", type=float,
                        help='Percentage of the Dataset for Training set',
                        default=.6)
    parser.add_argument("--validation_percentage", type=float,
                        help='Percentage of the Dataset for Validation set',
                        default=.2)
    parser.add_argument("--test_percentage", type=float,
                        help='Percentage of the Dataset for Test set',
                        default=.2)
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument("--only_stage", type=str,
                        help='Sleep stage to be considered for the analysis i.e., W, N1, N2, N3, REM. If None, no '
                             'distinction is made',
                        default=None)
    parser.add_argument("--only_brain_region", type=str,
                        help='Brain region to be considered for the analysis i.e., Fp, F, C, T, P, O. If None, no '
                             'distinction is made',
                        default=None)
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument("--run_pca", type=bool,
                        help='Flag to run dimensionality reduction (PCA)',
                        default=False)
    parser.add_argument("--explained_variance", type=float,
                        help='Percentage of data variance to explain (number of PCA components taken consequently)',
                        default=.95)
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument("--run_classification_task", type=bool,
                        help='Flag to run classification task',
                        default=False)

    return parser.parse_args()
