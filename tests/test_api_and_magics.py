# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL FREE SOFTWARE LICENSE AGREEMENT (Version 2.1)
# See full LICENSE agreement in the root directory
# =============================================================================


def test_api():
    import spectrochempy
    # test version
    from spectrochempy.api import APIref, log
    from spectrochempy import __version__ as version
    assert version.split('.')[0] == '0'
    assert version.split('.')[1][:1] == '1'
    # TODO: modify this for each release

    # test application

    print(('\n\nRunning : ', spectrochempy.api.running))
    assert version.startswith('0.1')

    log.warning('Ok, this is nicely executing!')

    assert 'np' in APIref
    assert 'NDDataset' in APIref
    assert 'abs' in APIref

def test_magic_addscript(ip):

    assert "available_styles" in ip.user_ns.keys()
    ip.run_cell("print(available_styles())", store_history=True)
    ip.run_cell("project = Project()", store_history=True)
    x = ip.magic('addscript -p project -o style -n available_styles 1')
                    # script with the definition of the function
                    # `available_styles` content of cell 2

    print("x", x)
    assert x.strip() == 'Script style created.'

    # with cell contents
    x = ip.run_cell('%%%%addscript -p project -o essai -n available_styles\n'
                    'print(available_styles())')

    print('result\n',x.result)
    assert x.result.strip() == 'Script essai created.'
