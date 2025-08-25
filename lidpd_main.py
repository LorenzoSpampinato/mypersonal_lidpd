import os
import sys
import time
import warnings
from data_loading import EEGDataLoader
from preprocess import EEGPreprocessor
from utilities import HypnogramProcessor
from feature_extraction import EEGFeatureExtractor
from utilities import EEGDataFrameGenerator
from utilities import NaNChecker
from utilities import DimensionalityReducer
from data_processing import DatasetSplitter
from model_training import Classifier
from input_script import get_args
import mne

warnings.filterwarnings('ignore')
args = get_args()  # Call get_args() from the imported file
# -----------------------------------------------  Dataset description  ------------------------------------------------
# - high-density EEG data from 257 channels
# - PNS (peripheral nervous system) data from 6 channels (i.e., Chin, Resp. Pressure, ECG, Resp. Effort,
#   Right and Left leg)


if __name__ == "__main__":

    # Defining the folder where to save the results
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # Save all prints in logfile
    log_dir = os.path.join(args.info_path, 'lorenzo_logfiles')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'logfile_{int(time.time())}.txt')
    sys.stdout = open(log_file_path, 'w')
    print("Logfile created")

    # ------------------------------------  Hypnogram definition (30-s epochs)  ------------------------------------
    processor = HypnogramProcessor(args.label_path, args.run_hypnogram)
    processor.process_hypnograms()

    # Definisci le classi di pazienti da elaborare
    classes = args.only_class.split(',') if args.only_class else ['DYS', 'ADV', 'DNV', 'CTL']

    # Loop per ogni classe
    for class_name in classes:
        print(f"Processing class: {class_name}")

        sub_folds = []
        class_folder = os.path.join(args.data_path, class_name)

        # Verifica se la cartella della classe esiste
        if not os.path.isdir(class_folder):
            print(f"Class folder {class_folder} does not exist!")
            continue  # Salta questa classe se la cartella non esiste

        # Se è specificato --only_patient, elaboriamo solo quei pazienti
        if args.only_patient:
            if args.only_patient == "ALL":  # Se è specificato "ALL" per i pazienti
                # Prendi tutti i pazienti dalla cartella della classe
                sub_folds = [folder for folder in os.listdir(class_folder)
                             if os.path.isdir(os.path.join(class_folder, folder))]
                print(f'Processing all patients in class {class_name}:', sub_folds)
            else:  # Altrimenti, prendi solo i pazienti specificati in args.only_patient
                patients = args.only_patient.split(',')
                for patient in patients:
                    patient_folder = os.path.join(class_folder, patient)
                    if os.path.isdir(patient_folder):  # Verifica che la cartella del paziente esista
                        sub_folds.append(patient)
                    else:
                        print(f"Patient folder {patient_folder} does not exist!")
        else:  # Se non è specificato --only_patient, prendi tutti i pazienti dalla cartella della classe
            sub_folds = [folder for folder in os.listdir(class_folder)
                         if os.path.isdir(os.path.join(class_folder, folder))]
            print(f'Processing all patients in class {class_name}:', sub_folds)

        # Elaborazione dei pazienti per la classe corrente
        for sub_fold in sub_folds:
            try:
                print(f"Start processing: {sub_fold}")

                # Definisci il percorso completo della cartella del paziente
                subject_path = os.path.join(class_folder, sub_fold)
                ###
                '''
                # Check se la cartella termina con '.mff' e definisci mff_path di conseguenza
                mff_path = subject_path if subject_path.endswith('.mff') else subject_path + '.mff'

                # Rinomina a .mff se necessario
                if not subject_path.endswith('.mff') and os.path.exists(subject_path):
                    print(f"Renaming {subject_path} to {mff_path}...")
                    os.rename(subject_path, mff_path)
                '''
                ###

                # --------------------------------------- Carica i dati grezzi --------------------------------------------------
                data_loader = EEGDataLoader(args.data_path, args.save_path, class_name, sub_fold)
                print("Save path:", args.save_path)
                #
                #raw = data_loader.load_and_prepare_data(mff_path, args.save_path, sub_fold)
                raw = data_loader.load_and_prepare_data(subject_path, args.save_path, sub_fold)

                # --------------------------------------- Preprocessing -----------------------------------------------------
                if args.run_preprocess or args.run_bad_interpolation:
                    preprocessor = EEGPreprocessor(
                        raw, args.data_path, args.label_path, args.save_path, args.run_preprocess,
                        args.run_bad_interpolation, class_name, sub_fold
                    )
                    preprocessed_raw = preprocessor.preprocess(sub_fold, overwrite=True)
                else:
                    preprocessed_raw = raw  # Usa i dati grezzi se non è specificato alcun preprocessing

                # ------------------------- Segmentazione in epoche ed estrazione delle caratteristiche ------------------------
                if args.run_features:
                    feature_extractor = EEGFeatureExtractor(
                        args.data_path, args.label_path, args.save_path, class_name, args.only_class, sub_fold
                    )
                    #
                    feature_extractor.process_and_save_features(subject_path, preprocessed_raw)
                    #feature_extractor.process_and_save_features(mff_path, preprocessed_raw)

                # --------------------------------  Aggregating labels + Dataframe definition  ---------------------------------
                if args.run_aggregation:
                    generator = EEGDataFrameGenerator(
                        args.label_path, args.save_path, args.aggregate_labels, class_name, sub_fold
                    )
                    generator.generate_dataframe(run_aggregation=args.run_aggregation)
                    print(f"Dataframe generated for patient: {sub_fold}")

                print(f"Finished processing: {sub_fold}")

            except Exception as e:
                print(f"Error encountered during processing of {sub_fold}: {e}")
            ###
            '''
            finally:
                # Ripristina il nome originale della cartella se necessario
                if os.path.exists(mff_path):
                    print(f"Renaming {mff_path} back to {subject_path}...")
                    os.rename(mff_path, subject_path)
                    print(f"Restored original name: {subject_path}")
            '''

            ###

    # -------------------------------------------  Checking NaN values  --------------------------------------------
    checker = NaNChecker(args.save_path, args.run_nan_check)
    checker.check_nan_values()
    # ------------------------------  Training, Validation, and Test sets definition  ------------------------------
    splitter = DatasetSplitter(args.save_path, args.training_percentage,
                               args.validation_percentage, args.test_percentage)
    splitter.train_val_test_split(args.run_dataset_split)

    # ----------------------------------------  Dimensionality reduction  ------------------------------------------
    reducer = DimensionalityReducer(args.save_path, args.run_pca, args.explained_variance,
                                    args.only_stage, args.only_brain_region)
    reducer.run()

    # -------------------------------------------  Classification task  --------------------------------------------
    classifier = Classifier(args.save_path, args.only_stage, args.only_brain_region)
    classifier.classification(args.run_classification_task, args.run_pca)