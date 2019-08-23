# -*- coding: utf-8 -*-

class WholeView():
    def __init__(self, train, test=None, numcols=None, catcols=None):
        if not isinstance(train, pd.DataFrame):
            raise('train is not pd.DataFrame')

        if test is not None:
            if not isinstance(test, pd.DataFrame):
                raise('test is not pd.DataFrame')

        self.train = train
        self.test = test
        self.numcols = numcols
        self.catcols = catcols
    
    def _cols(self):
        if self.numcols is None:
            self.numcols = [c for c in self.train.columns if self.train[c].dtype != 'object']
        
        if self.catcols is None:
            self.catcols = [c for c in self.train.columns if self.train[c].dtype == 'object']
        
    def _numfunc(self, c, data):
        self.numstats.setdefault('count', list()).append(data.shape[0])
        self.numstats.setdefault('missing', list()).append('%.2f%%' % (np.sum(np.isnan(data[c].values)) / data.shape[0] * 100))
        self.numstats.setdefault('mean', list()).append(np.mean(data[c].values))
        self.numstats.setdefault('std', list()).append(np.std(data[c].values))
        self.numstats.setdefault('zeros / notnull', list()).append('%.2f%%' % (np.sum(np.where(data[c] == 0, 1, 0)) / np.sum(~np.isnan(data[c].values)) * 100))
        self.numstats.setdefault('min', list()).append(np.min(data[c].values))
        self.numstats.setdefault('median', list()).append(np.median(data[c].values))
        self.numstats.setdefault('max', list()).append(np.max(data[c].values))

    def _catfunc(self, c, data):
        self.catstats.setdefault('count', list()).append(data.shape[0])
        self.catstats.setdefault('missing', list()).append('%.2f%%' % (data[c].isnull().sum() / data.shape[0] * 100))
        self.catstats.setdefault('unique', list()).append(data.loc[data[c].notnull(), c].nunique())
        self.catstats.setdefault('top', list()).append(data[c].mode().values[0])
        self.catstats.setdefault('freq top / notnull', list()).append('%.2f%%' % ((data[c] == data[c].mode().values[0]).sum() / data[c].notnull().sum() * 100))

    def _numstats(self, n_jobs):
        self.numstats = {}
        # if self.test is not None:
        #     data = 
        # self.numstats['count'] = np.zeros(len(self.numcols)) * self.train.shape[0]
        # self.numstats['missing'] = [('%.2f%%' % (i / self.train.shape[0] * 100)) for i in np.sum(np.isnan(self.train[self.numcols].values), axis=0)]
        # self.numstats['mean'] = np.mean(self.train[self.numcols].values, axis=0)
        # self.numstats['std'] = np.std(self.train[self.numcols].values, axis=0)
        # self.numstats['zeros / notnull'] = [('%.2f%%' % (i * 100)) 
        #                                     for i in np.sum(np.where(self.train[self.numcols] == 0, 1, 0), axis=0) / 
        #                                     np.sum(~np.isnan(self.train[self.numcols].values), axis=0)]
        # self.numstats['min'] = np.min(self.train[self.numcols].values, axis=0)
        # self.numstats['median'] = np.median(self.train[self.numcols].values, axis=0)
        # self.numstats['max'] = np.max(self.train[self.numcols].values, axis=0)

        parallel(n_jobs)(delayed(self._numfunc)(c, self.train) for c in self.numcols)
        if self.test is not None:
            parallel(n_jobs)(delayed(self._numfunc)(c, self.test) for c in self.numcols)

    def _catstats(self, n_jobs):
        self.catstats = {}
        parallel(n_jobs)(delayed(self._catfunc)(c, self.train) for c in self.catcols)
        if self.test is not None:
            parallel(n_jobs)(delayed(self._catfunc)(c, self.test) for c in self.catcols)

    def showstats(self, n_jobs=1):
        self._cols()
        # numstat
        self._numstats(n_jobs)
        if self.test is not None:
            num_stat = pd.DataFrame(self.numstats, index=pd.MultiIndex.from_product([self.numcols, ['train', 'test']]))
        else:
            num_stat = pd.DataFrame(self.numstats, index=self.numcols)
        # catstat
        self._catstats(n_jobs)
        if self.test is not None:
            cat_stat = pd.DataFrame(self.catstats, index=pd.MultiIndex.from_product([self.catcols, ['train', 'test']]))
        else:
            cat_stat = pd.DataFrame(self.catstats, index=self.catcols)
        return num_stat, cat_stat
    
    def plotnums(self, cols, bins, ylim=None, ytick=None, figsize=(8, 4)):
        for c in cols:
            plt.figure(figsize=figsize)
            try:
                plt.hist([self.train[c].values, self.test[c].values], bins=bins, stacked=True)
                plt.legend(['train', 'test'])
                plt.title('column %s hist of train and test' % c)
                plt.ylim(ylim)
                plt.yticks = ytick
            except:
                plt.hist(self.train[c].values, bins=bins)
                plt.legend(['train'])
                plt.title('column %s hist of train' % c)
                plt.ylim(ylim)
                plt.yticks = ytick
        plt.show()

    def plotcats(self, cols, bins, ylim=None, ytick=None, figsize=(8, 4)):
        for c in cols:
            plt.figure(figsize=figsize)
            try:
                self.train['cate'] = 'train'
                self.test['cate'] = 'test'
                data = pd.concat([self.train, self.test], axis=0)
                data.groupby([c, 'cate'])['cate'].count().unstack().fillna(0).plot(kind='bar', stacked=True)
                # plt.legend(['train', 'test'])
                plt.title('column %s count of train and test' % c)
                plt.ylim(ylim)
                plt.yticks = ytick
            except:
                plt.bar(self.train[c].value_counts.index, self.train[c].value_counts)
                plt.legend(['train'])
                plt.title('column %s count of train' % c)
                plt.ylim(ylim)
                plt.yticks = ytick
        plt.show()