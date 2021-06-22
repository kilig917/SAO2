from Weight import Weight
import numpy as np

patent_test = [{'US20020010638A1': '^_^(cyber physical system, performing, );'
                                   '^_^(cyber physical system, performing, );^_^(, comprising, academic user)',
                'US20040236640A1': '^_^(cyber physical system, printed, cyber physical system);'
                                   '^_^(, providing, aniline molecularly imprinted)'}]
SAOList = {'US20020010638A1': [['cyber physical system', 'cyber physical system', '', '', '', '', '', '',
                                'cyber physical system', 'cyber physical system', 'cyber physical system',
                                'cyber physical system', 'cyber physical system', 'cyber physical system',
                                'cyber physical system', 'cyber physical system'],
                               ['performing', 'performing', 'comprising', 'coupled', 'comprising', 'comprising',
                                'coupled', 'comprising', 'submitting', 'receiving', 'establishing', 'review',
                                'approve', 'enabling', 'review', 'approve'],
                               ['', '', 'academic user', 'cyber physical system', 'cyber physical system',
                                'academic user', 'cyber physical system', 'cyber physical system',
                                'access lists', 'access lists', 'times timesn fiber connection', '', '',
                                'access times timesn fiber connection', '', '']],
           'US20040236640A1': [['cyber physical system', '', '', 'accounting software', 'accounting software',
                                'cyber physical system', 'accounting software', 'accounting software',
                                'cyber physical system', '', '', 'access', 'access', 'accounting software',
                                'accounting software', 'cyber physical system', 'accounting software',
                                'accounting software', 'web functional composites', 'cyber physical system',
                                'accounting software', 'accounting software', 'cyber physical system',
                                'accounting software', 'accounting software', 'accounting software', 'access',
                                'web functional composites', 'access', 'accounting software',
                                'accounting software', 'cyber physical system', 'accounting software',
                                'accounting software', 'accounting software', 'cyber physical system',
                                'cyber physical system', 'cyber physical system',
                                'acoustoelectronic imaging production', 'acoustoelectronic imaging production',
                                'cyber physical system', 'acoustoelectronic imaging production',
                                'acoustoelectronic imaging production', 'cyber physical system',
                                'requested activation web', 'delivery visitor', 'cyber physical system',
                                'requested activation web', 'formyldihydropteroylglutamic acid'],
                               ['printed', 'providing', 'printed', 'enabling', 'submit', 'printed', 'receiving',
                                'printed', 'printed', 'providing', 'printed', 'received', 'comprising',
                                'enabling', 'submit', 'printed', 'receiving', 'printed', 'add', 'add',
                                'enabling', 'submit', 'printed', 'receiving', 'printed', 'hosted', 'providing',
                                'add', 'add', 'enabling', 'submit', 'printed', 'receiving', 'printed', 'hosted',
                                'receiving', 'representing', 'processing', 'representing', 'fulfilling',
                                'provide', 'representing', 'fulfilling', 'transmitting', 'hosted', 'printed',
                                'providing', 'hosted', 'printed'],
                               ['cyber physical system', 'aniline molecularly imprinted',
                                'cyber physical system', 'delivery visitor', 'access request',
                                'cyber physical system', 'xe', 'cyber physical system', 'cyber physical system',
                                'aniline molecularly imprinted', 'cyber physical system',
                                'communication network', 'cyber physical system', 'delivery visitor',
                                'access request', 'cyber physical system', 'xe', 'cyber physical system',
                                'cyber physical system', 'web addition', 'delivery visitor', 'access request',
                                'cyber physical system', 'xe', 'cyber physical system', '',
                                'web addition web access xe web web', 'cyber physical system', 'web addition',
                                'delivery visitor', 'access request', 'cyber physical system', 'xe',
                                'cyber physical system', '', 'acoustoelectronic imaging', 'access request', '',
                                'instruction', 'access request', 'acoustoelectronic imaging production',
                                'instruction', 'access request', 'production', '', 'cyber physical system',
                                 'web data application administration acid', '', 'cyber physical system']]}

weightTestSys = Weight(patent_test, patent_test[0], [[1] * 100] * 20, SAOList)
weightTestSys.Graph_set_up()