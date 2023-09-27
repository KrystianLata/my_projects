import pandas as pd


class FeatureExtractor:

    def __init__(self, config):
        self.config = config
    
    def process_id_col(self, df):
        df[['GroupNum', 'NumInGroup']] = df['PassengerId'].str.split('_', expand=True)
        df['NumInGroup'] = df['NumInGroup'].astype(int)
        df['GroupSize'] = df.groupby('GroupNum')['NumInGroup'].transform('max')
        df = df.drop(columns=['PassengerId', 'NumInGroup'])
        
        return df
    
    def process_cabin_col(self, df):
        df[['CabinDeck', 'CabinNum', 'CabinSide']] = df['Cabin'].str.split('/', expand=True)
        df['CabinNum'] = df['CabinNum'].astype(float)
        df = df.drop(columns=['Cabin'])

        return df
    
    def process_name_col(self, df):
        df['Surname'] = df['Name'].str.split(' ').str[-1]
        df = df.drop(columns=['Name'])

        return df
    
    def process_alone(self, df):
        df['Alone'] = df['GroupSize'] == 1
    
        return df
    
    def process_family(self, df):
        """ Function checks if passenger is travelling with family (basing on surname and group members)"""
        
        # find number of group members with the same surname
        group_counts = df.groupby(['GroupNum', 'Surname']).size().reset_index(name='SameGroupSurnameCount')
        df = df.merge(group_counts, on=['GroupNum', 'Surname'], how='left')
        df['WithFamily'] = df['SameGroupSurnameCount'] > 1
        
        # remove temp columns
        df = df.drop(columns=['SameGroupSurnameCount', 'Surname'])

        return df
    
    def set_datatypes(self, df):
        # df['HomePlanet'] = df['HomePlanet'].astype('category')
        df['CryoSleep'] = df['CryoSleep'].astype(float)
        # df['Destination'] = df['Destination'].astype('category')
        df['Age'] = df['Age'].astype(float)
        df['VIP'] = df['VIP'].astype(float)
        df['RoomService'] = df['RoomService'].astype(float)
        df['FoodCourt'] = df['FoodCourt'].astype(float)
        df['ShoppingMall'] = df['ShoppingMall'].astype(float)
        df['Spa'] = df['Spa'].astype(float)
        df['VRDeck'] = df['VRDeck'].astype(float)
        df['GroupNum'] = df['GroupNum'].astype(float)
        df['GroupSize'] = df['GroupSize'].astype(float)
        # df['CabinDeck'] = df['CabinDeck'].astype('category')
        df['CabinNum'] = df['CabinNum'].astype(float)
        # df['CabinSide'] = df['CabinSide'].astype('category')
        df['Alone'] = df['Alone'].astype(float)
        df['WithFamily'] = df['WithFamily'].astype(float)

        return df
    
    def drop_columns(self, df):
        for col in df.columns:
            if col not in self.config['COLUMNS_LIST']:
                df = df.drop(columns=[col])

                print(f'Dropped column: {col}')

        return df

    def extract(self, df):
        """Function combines all extracting subfunctions."""
        df = df.replace({True:1, False:0})
        df = self.process_id_col(df)
        df = self.process_cabin_col(df)
        df = self.process_name_col(df)
        df = self.process_alone(df)
        df = self.process_family(df)
        df = self.set_datatypes(df)
        df = self.drop_columns(df)

        return df

class ManualNaFiller:
    def __init__(self, config):
        self.config = config
        self.log_list = []

    def _missing_data_summary(self, df):
        total = df.isnull().sum().sort_values(ascending=False)
        percent = (df.isnull().sum() / df.isnull().count() * 100).sort_values(ascending=False)
        return pd.concat([total, percent], axis=1, keys=['Total Missing', 'Percent (%)'])

    def _cryosleep_from_services(self, df):
        services_cols = ['ShoppingMall', 'Spa', 'VRDeck', 'FoodCourt', 'RoomService']
        before_na_count = df['CryoSleep'].isnull().sum()

        mask = df['CryoSleep'].isnull()
    
        df.loc[mask, 'CryoSleep'] = df.loc[mask, services_cols].gt(0).any(axis=1)

        after_na_count = df['CryoSleep'].isnull().sum()

        self.log_list.append({'action': 'CRYOSLEEP_FROM_SERVICES', 
                              'filled_values_num': before_na_count - after_na_count, 
                              'remaining_na_in_df' : df.isnull().sum().sum()})
        
        return df
    
    def _services_from_cryosleep(self, df):
        services_cols = ['ShoppingMall', 'Spa', 'VRDeck', 'FoodCourt', 'RoomService']
        before_na_count = sum(df[col].isnull().sum() for col in services_cols)

        mask = df['CryoSleep'] == 1
        
        # Update services_cols where 'CryoSleep' is True
        df.loc[mask, services_cols] = 0

        after_na_count = sum(df[col].isnull().sum() for col in services_cols)
        
        self.log_list.append({'action': 'SERVICES_FROM_CRYOSLEEP', 
                              'filled_values_num': before_na_count - after_na_count, 
                              'remaining_na_in_df' : df.isnull().sum().sum()})
        return df

    def _vip_from_age_threshold(self, df):
        treshold = self.config['VIP_FROM_AGE_TRESHOLD']

        # Mask for missing VIP values and age less than threshold
        mask_less_than_threshold = df['VIP'].isnull() & (df['Age'] < treshold)
        mask_greater_than_threshold = df['VIP'].isnull() & (df['Age'] >= treshold)

        # Update VIP based on conditions
        df.loc[mask_less_than_threshold, 'VIP'] = 0
        df.loc[mask_greater_than_threshold, 'VIP'] = 1

        # Log the changes
        filled_values_num = mask_less_than_threshold.sum() + mask_greater_than_threshold.sum()

        self.log_list.append({
            'action': 'VIP_FROM_AGE_TRESHOLD',
            'filled_values_num': filled_values_num,
            'remaining_na_in_df': df.isnull().sum().sum()
        })
        return df

    def na_fill(self, df):

        if self.config['CRYOSLEEP_FROM_SERVICES']:
            df = self._cryosleep_from_services(df)

        if self.config['SERVICES_FROM_CRYOSLEEP']:
            df = self._services_from_cryosleep(df)

        if self.config['VIP_FROM_AGE_TRESHOLD']:
            df = self._vip_from_age_threshold(df)

        print('\nna fill log:\n')
        na_fill_log = pd.DataFrame(self.log_list)
        print(na_fill_log)

        print(f'\nMissing summary:\n\n{self._missing_data_summary(df)}')

        
        self.log_list = []
        return df


class Preprocessor:

    def __init__(self, config):
        self.config = config
        self.random_state = config['MAIN_CONFIG']['RANDOM_STATE']
        self.feature_extractor = FeatureExtractor(config['PREPROCESS_CONFIG'])
        self.manual_na_filler = ManualNaFiller(config['PREPROCESS_CONFIG'])

    def process(self, df):
        print(f'df input shape: {df.shape}')

        # extracting features
        df = self.feature_extractor.extract(df)

        # manual fill missing values
        df = self.manual_na_filler.na_fill(df)

        print(f'df after preprocessing shape: {df.shape}')
        return df