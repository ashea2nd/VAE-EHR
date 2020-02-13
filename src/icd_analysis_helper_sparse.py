from typing import List, Tuple

import numpy as np 
import pandas as pd

class ICDAnalysisHelperSparse:
    # icd9codes_df = None
    # patient_icd_df = None

    # def __init__(self,
    #     icd9codes_df: pd.DataFrame,
    #     patient_icd_df: pd.DataFrame):

    #     self.icd9codes_df = icd9codes_df
    #     self.patient_icd_df = patient_icd_df

    #     ICDAnalysisHelper.icd9codes_df = icd9codes_df
    #     ICDAnalysisHelper.patient_icd_df = patient_icd_df

    # #Used for DISEASE clustering
    # def get_icd_idxs(
    #     self,
    #     substrings: List[str], 
    #     case_sensitive: bool=True, 
    #     verbose: bool=False):
        
    #     idxs = []
    #     for substring in substrings:
    #         icds_with_substring = self.icd9codes_df[self.icd9codes_df.LONG_TITLE.str.contains(substring, case=case_sensitive)].ICD9_CODE.tolist()
    #         print("Occurences of {0} before filter: {1}".format(substring, len(icds_with_substring)))
            
    #         icds_with_substring_and_in_patient_icd_df = [icd for icd in icds_with_substring if icd in self.patient_icd_df.columns]
    #         icds_with_substring=icds_with_substring_and_in_patient_icd_df
    #         print("After:", len(icds_with_substring))

    #         idx = [self.patient_icd_df.columns.get_loc(icd) - 1 for icd in icds_with_substring]
    #         idxs += idx
        
    #     if verbose:
    #         print(self.icd9codes_df[self.icd9codes_df.SHORT_TITLE.str.contains(substring, case=case_sensitive)])
    #     print("Total found: {}".format(len(idxs)))
    #     return idxs

    # #Use for PATIENT clustering
    # def get_patients_idxs_with_disease_keywords(
    #     self,
    #     patient_df: pd.DataFrame,
    #     substrings: List[str],
    #     case_sensitive: bool=False,
    #     ):
        
    #     idxs = []
    #     relevant_icds = []
    #     patient_df = patient_df.reset_index()
        
    #     for substring in substrings:
    #         icds_with_substring = self.icd9codes_df[self.icd9codes_df.LONG_TITLE.str.contains(substring, case=case_sensitive)].ICD9_CODE.tolist()
    #         print("Occurences of {0} before filter: {1}".format(substring, len(icds_with_substring)))

    #         icds_with_substring_and_in_patient_icd_df = [icd for icd in icds_with_substring if icd in patient_df.columns]
    #         icds_with_substring=icds_with_substring_and_in_patient_icd_df        
    #         print("After:", len(icds_with_substring))
            
    #         relevant_icds += icds_with_substring_and_in_patient_icd_df
            
    #     print("Total Relevant ICDs: {}".format(len(relevant_icds)))
    #     patients_with_disease = patient_df.loc[:, relevant_icds].any(axis=1)
    #     patients_with_disease = patients_with_disease[patients_with_disease == True]
    #     print("Patients with disease(s): {}".format(len(patients_with_disease)))
        
    #     return patients_with_disease.index.tolist()

    # @classmethod
    # def most_common_diseases_in_cohort(
    #     cls,
    #     patient_idxs: List[int]
    #     ):
    #     patients_of_interest = cls.patient_icd_df.drop('SUBJECT_ID', axis=1).iloc[patient_idxs]
    #     disease_sums = patients_of_interest.sum(axis=0)
    #     return disease_sums.sort_values(ascending=False)

    # @classmethod
    # def lookup_icds(
    #     cls,
    #     icd9codes_list: List[str]
    #     ):
    #     return cls.icd9codes_df[cls.icd9codes_df['ICD9_CODE'].isin(icd9codes_list)]

    # @classmethod
    # def get_icd_short_titles(
    #     cls,
    #     icd9codes_list: List[str],
    #     verbose=True
    #     ):
    #     short_titles = cls.icd9codes_df[cls.icd9codes_df['ICD9_CODE'].isin(icd9codes_list)].SHORT_TITLE.values
    #     if len(short_titles) > 1:
    #         return short_titles
    #     elif len(short_titles) == 1:
    #         return short_titles[0]
    #     else:
    #         if verbose:
    #             print("No short title found for ICD code {}".format(icd9codes_list))
    #         return icd9codes_list[0]

    # @classmethod
    # def get_icd_long_titles(
    #     cls,
    #     icd9codes_list: List[str],
    #     verbose=True
    #     ):
    #     long_titles = cls.icd9codes_df[cls.icd9codes_df['ICD9_CODE'].isin(icd9codes_list)].LONG_TITLE.values
    #     if len(long_titles) > 1:
    #         return long_titles
    #     elif len(long_titles) == 1:
    #         return long_titles[0]
    #     else:
    #         if verbose:
    #             print("No long title found for ICD code {}".format(icd9codes_list))
    #         return icd9codes_list[0]