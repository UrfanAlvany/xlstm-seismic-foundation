import numpy as np
from obspy import Stream
from obspy.core.inventory import Inventory



def full_preprocessing(
        st: Stream, 
        remove: str = 'response', 
        output: str = 'ACC', 
        rotate: bool = False, 
        inv: Inventory = None,
        pre_filt: tuple = (1. / 50., 1. / 30., 9., 9.3),
        ) -> Stream:
    """
    Preprocessing function for seismic data. Should be applied to the full data stream.
    This is not meant to preprocess event traces.
    
    - detrend data: st.detrend('constant')
    - optionally remove sensitivity or response
    - rotate to ZNE, if not already in that format
    - get zeroes back into gaps


    Parameters
    ----------
    :param st: Stream
        The seismic data stream to preprocess. Either in ZNE or UVW format.
        Usually at 20Hz sampling rate.
    :param remove: str, optional
        The type of preprocessing to apply. Options are 'response'. 'sensitivity' or 'None'.
        'response' removes the instrument response, 'sensitivity' removes the sensitivity,
        and 'None' does not apply any removal.
        Default is 'response'.
    :param output: str, optional
        The output type of the data. Options are 'DISP', 'ACC', 'VEL' or 'None'
        Default is 'ACC'.
    :param rotate: bool, optional
        Whether to rotate the data to ZNE format. Default is False.
    :param inv: Inventory, optional
        The inventory containing station metadata. Default is None.
    :param pre_filt: tuple, optional
        The pre-filtering frequencies for response removal. Default is (1. / 50., 1. / 30., 9., 9.3).
    Returns
    -------
    :return: Stream
        The preprocessed seismic data stream.
    """

    if remove not in ['response', 'sensitivity', 'None']:
        raise ValueError("remove must be either 'response', 'sensitivity' or 'None'")
    if output not in ['DISP', 'ACC', 'VEL', 'None']:
        raise ValueError("output must be either 'DISP', 'ACC', 'VEL' or 'None'")
    if inv is None:
        raise ValueError("Inventory must be provided for response removal or rotation")    

    # save gaps for later
    gaps = st.get_gaps()

    # merge stream (probably not needed, but just in case)
    st.merge(method=1, fill_value=0)

    # detrend data
    st.detrend('constant')

    if remove == 'response':
        # remove instrument response
        st.remove_response(inventory=inv, output=output, pre_filt=pre_filt)
        print("Removed instrument response")
    elif remove == 'sensitivity':
        # remove sensitivity
        st.remove_sensitivity(inventory=inv)
        print("Removed sensitivity")
    else:
        print("No response or sensitivity removal applied")


    if rotate:
        st.rotate(method='->ZNE', inventory=inv, components=['UVW'])
        # sort traces by component (Z, N, E)
        st.sort(['component'], reverse=True)  
        print("Rotated to ZNE format")
    else:
        print("No rotation applied")

    # reinsert gaps and fill with zeroes
    if len(gaps) > 0 and (rotate or remove != 'None'):
        for i in range(len(gaps)):
            # cut out gaps
            st.cutout(gaps[i][4], gaps[i][5])
        # fill gaps with zeroes
        st.merge(method=0, fill_value=0) # method=0 discards overlapping data and fills gaps with zeros

    # st should now have dtype float64
    # TODO: change dtype to float32 if needed
    for trace in st:
        trace.data = trace.data.astype(np.float32)
        
    print("Preprocessing complete")
    return st