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
        tmp = {}
        tmp['count'] = data.shape[0]
        tmp['missing'] = '%.2f%%' % (np.sum(np.isnan(data[c].values)) / data.shape[0] * 100)
        tmp['mean'] = np.mean(data[c].values)
        tmp['std'] = np.std(data[c].values)
        tmp['zeros / notnull'] = '%.2f%%' % (np.sum(np.where(data[c] == 0, 1, 0)) / np.sum(~np.isnan(data[c].values)) * 100)
        tmp['min'] = np.min(data[c].values)
        tmp['median'] = np.median(data[c].values)
        tmp['max'] = np.max(data[c].values)
        return tmp

    def _catfunc(self, c, data):
        tmp = {}
        tmp['count'] = data.shape[0]
        tmp['missing'] = '%.2f%%' % (data[c].isnull().sum() / data.shape[0] * 100)
        tmp['unique'] = data.loc[data[c].notnull(), c].nunique()
        tmp['top'] = data[c].mode().values[0]
        tmp['freq top / notnull'] = '%.2f%%' % ((data[c] == data[c].mode().values[0]).sum() / data[c].notnull().sum() * 100)
        return tmp

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

        numlist = Parallel(n_jobs)(delayed(self._numfunc)(c, self.train) for c in self.numcols)

        if self.test is not None:
            numlist += Parallel(n_jobs)(delayed(self._numfunc)(c, self.test) for c in self.numcols)
        return numlist

    def _catstats(self, n_jobs):
        self.catstats = {}
        catlist = Parallel(n_jobs)(delayed(self._catfunc)(c, self.train) for c in self.catcols)
        if self.test is not None:
            catlist += Parallel(n_jobs)(delayed(self._catfunc)(c, self.test) for c in self.catcols)
        return catlist

    def showstats(self, n_jobs=1):
        self._cols()
        # numstat
        m = len(self.numcols)
        numlist = self._numstats(n_jobs)
        if self.test is not None:
            for idx in range(m):
                self.numstats.setdefault('count', list()).append(numlist[idx]['count'])
                self.numstats.setdefault('count', list()).append(numlist[idx + m]['count'])
                self.numstats.setdefault('missing', list()).append(numlist[idx]['missing'])
                self.numstats.setdefault('missing', list()).append(numlist[idx + m]['missing'])
                self.numstats.setdefault('mean', list()).append(numlist[idx]['mean'])
                self.numstats.setdefault('mean', list()).append(numlist[idx + m]['mean'])
                self.numstats.setdefault('std', list()).append(numlist[idx]['std'])
                self.numstats.setdefault('std', list()).append(numlist[idx + m]['std'])
                self.numstats.setdefault('zeros / notnull', list()).append(numlist[idx]['zeros / notnull'])
                self.numstats.setdefault('zeros / notnull', list()).append(numlist[idx + m]['zeros / notnull'])
                self.numstats.setdefault('min', list()).append(numlist[idx]['min'])
                self.numstats.setdefault('min', list()).append(numlist[idx + m]['min'])
                self.numstats.setdefault('median', list()).append(numlist[idx]['median'])
                self.numstats.setdefault('median', list()).append(numlist[idx + m]['median'])
                self.numstats.setdefault('max', list()).append(numlist[idx]['max'])
                self.numstats.setdefault('max', list()).append(numlist[idx + m]['max'])        
            num_stat = pd.DataFrame(self.numstats, index=pd.MultiIndex.from_product([self.numcols, ['train', 'test']]))
        else:
            for idx in range(m):
                self.numstats.setdefault('count', list()).append(numlist[idx]['count'])
                self.numstats.setdefault('missing', list()).append(numlist[idx]['missing'])
                self.numstats.setdefault('mean', list()).append(numlist[idx]['mean'])
                self.numstats.setdefault('std', list()).append(numlist[idx]['std'])
                self.numstats.setdefault('zeros / notnull', list()).append(numlist[idx]['zeros / notnull'])
                self.numstats.setdefault('min', list()).append(numlist[idx]['min'])
                self.numstats.setdefault('median', list()).append(numlist[idx]['median'])
                self.numstats.setdefault('max', list()).append(numlist[idx]['max'])
            num_stat = pd.DataFrame(self.numstats, index=self.numcols)

        # catstat
        n = len(self.catcols)
        catlist = self._catstats(n_jobs)
        if self.test is not None:
            for idx in range(n):
                self.catstats.setdefault('count', list()).append(catlist[idx]['count'])
                self.catstats.setdefault('count', list()).append(catlist[idx + n]['count'])
                self.catstats.setdefault('missing', list()).append(catlist[idx]['missing'])
                self.catstats.setdefault('missing', list()).append(catlist[idx + n]['missing'])
                self.catstats.setdefault('unique', list()).append(catlist[idx]['unique'])
                self.catstats.setdefault('unique', list()).append(catlist[idx + n]['unique'])
                self.catstats.setdefault('top', list()).append(catlist[idx]['top'])
                self.catstats.setdefault('top', list()).append(catlist[idx + n]['top'])
                self.catstats.setdefault('freq top / notnull', list()).append(catlist[idx]['freq top / notnull'])
                self.catstats.setdefault('freq top / notnull', list()).append(catlist[idx + n]['freq top / notnull'])
            cat_stat = pd.DataFrame(self.catstats, index=pd.MultiIndex.from_product([self.catcols, ['train', 'test']]))
        else:
            for idx in range(n):
                self.catstats.setdefault('count', list()).append(catlist[idx]['count'])
                self.catstats.setdefault('missing', list()).append(catlist[idx]['missing'])
                self.catstats.setdefault('unique', list()).append(catlist[idx]['unique'])
                self.catstats.setdefault('top', list()).append(catlist[idx]['top'])
                self.catstats.setdefault('freq top / notnull', list()).append(catlist[idx]['freq top / notnull'])
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