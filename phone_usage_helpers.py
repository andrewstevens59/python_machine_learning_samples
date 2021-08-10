# NOTICE: THIS PROGRAM CONSISTS OF TRADE SECRETS THAT ARE THE PROPERTY OF TRUEMOTION, INC. AND ARE PROPRIETARY MATERIAL OF TRUEMOTION, INC.
import numpy as np
import pandas as pd
import logging

from scipy.interpolate import interp1d
import copy

from Events.helpers import interval_helpers
logger = logging.getLogger('DI')

SPEED_CUTOFF = 2.68224

def dx_dt(x, time, resolution=10**-2):
    """
    Differentiates x w.r.t. time
    
    Args:
        x (array-like <float>): The input signal to differentiate
        time (array-like <float>): The reference time for the differentiation
        resolution (float, default=10**-2): The smallest period of time such that dx should be considered 0
    Returns:
        Derivative of x with respect to t
    """
    dx = np.diff(x)
    dt = np.diff(time)
    acc = dx/dt
    acc[dt <= resolution] = 0
    return acc

def diff_grav_acc(o_df):
    """
    Adds a column to the o_df with information on how frequently the phone changes position.
    The new column is named "diff_grav_acc", which is essentially the magnitude of the 
    gravity acceleration vector. 
    
    Args:
        o_df (pd.Dataframe): An "orientation" data frame.
    Returns:
        o_df (pd.Dataframe): Copy of the input, with the column "diff_grav_acc" added.
    """

    # Make sure the timestamp is the index of the inout data frame.
    o_df.index = o_df["time_unix_epoch"]

    # COmpute the magnitude of the acceleration vector.
    dgx = dx_dt(o_df["grav_x"], o_df["time_unix_epoch"])[...,None]
    dgy = dx_dt(o_df["grav_y"], o_df["time_unix_epoch"])[...,None] 
    dgz = dx_dt(o_df["grav_z"], o_df["time_unix_epoch"])[...,None]
    diffs = np.linalg.norm(
        np.hstack((dgx[1:], dgy[1:], dgz[1:])) - np.hstack((dgx[:-1],dgy[:-1],dgz[:-1])),
        axis=1
    )
    diffs = np.hstack((np.zeros(2), diffs))

    # Put the result back into the input data frame.
    o_df["diff_grav_acc"] = pd.Series(diffs, index=o_df.index)

    return o_df


def time_satisfied(df, condition):
    """
    Returns the amount of time that the condition on the data is satisfied
    
    Args:
        df (pd.Dataframe): A DataFrame that has a time_unix_epoch column
        condition (function): A boolean valued function on the data frame
        
    Returns:
        The amount of time that the condition is satisified
    
    """
    assert "time_unix_epoch" in df
    if len(df) < 2:
        out = 0
    else:
        dt = df["time_unix_epoch"].diff()
        dt[0] = 0
        out = float(sum(dt[condition(df)]))
    return out


def create_window(gps, start_window_time, end_window_time, flags=None, method=None):
    """
    This method creates a structure defining a segment of a GPS track. The structure includes
    the start and end times, along with the lon/lat values associated with those times.
    Ths routine optionally adds a mehtod name and a set of arbitrary key/value pairs.

    Args:
        gps (pd.DataFrame)
        start_window_time (float)
        end_window_time (float)
        flags (dict): Additional flags to add to the window
        method (str): The method used to create the window

    Returns:
        The window
    """
    method = "phone_usage" if method is None else method

    # Find the indices of the GPS records corresponding to the given start and end times.
    gps_start_ind = min(len(gps) - 1, gps["time_unix_epoch"].searchsorted(start_window_time)[0])
    gps_end_ind   = min(len(gps) - 1, gps["time_unix_epoch"].searchsorted(end_window_time)[0])

    D = {
                'method_name':  method,
                'start':        float(start_window_time),
                'end':          float(end_window_time),
                'startLat':     float(gps["latitude"].values[gps_start_ind]),
                'endLat':       float(gps["latitude"].values[gps_end_ind]),
                'startLong':    float(gps["longitude"].values[gps_start_ind]),
                'endLong':      float(gps["longitude"].values[gps_end_ind]),
                'startTz':      float(gps["time_zone"].values[gps_start_ind]),
                'endTz':        float(gps["time_zone"].values[gps_end_ind])
            }

    # Add optional flags.
    if not flags is None:
        for k,v in flags.items():
            D[k] = v

    return D
    

def consolidate_windows(windows, consolidation_size=180, window_type="dict", invalidated_segment_times = []):
    """
    Scans the windows array and consolidates nearby windows together. 2 windows are considered 
    "nearby" if they are within a consolidation size of each other

    Args:
        windows (list<dict>): List of dictionaries or arrays where each dictionary corresponds to an interval

        consolidation_size (int): The max distance between 2 windows for the windows to be consolidated
        
        invalidated_segment_times (int): A set of times corresponding to windows that should not be included in the set of consolidated windows


    Returns:
        The windows list with nearby windows consolidated
    """
    
    print invalidated_segment_times
    
    if not len(windows):
        out = [], []
    elif len(windows) == 1:
        out = windows, [[0]]
    else:
        prev_window = None
        consolidated = []
        indices = []
        ind = None
        
        invalid_time = -1
        invalid_time_offset = 0
        if len(invalidated_segment_times) > 0:
            invalid_time = invalidated_segment_times[0]
            invalid_time_offset += 1
            
        prev_window_time = 0

        for i, window in zip(range(len(windows)), windows):
            

            if prev_window != None:
                
                while invalid_time < prev_window['end']:
                    if invalid_time_offset >= len(invalidated_segment_times):
                        break

                    invalid_time = invalidated_segment_times[invalid_time_offset]
                    invalid_time_offset += 1
                    
               
                print invalid_time
                print prev_window
                
                if prev_window['end'] < invalid_time and invalid_time < window['start']:
                    print "here"
                    # First append the valid indices if previous window was valid

                    if ind != None:
                        indices.append(copy.deepcopy(ind))
                        consolidated.append(prev_window)

                    # Next reset the valid indices 
                    ind = None
                    prev_window = None
                    continue
     
            # If None, this is the first window in the sequence and it 
            # is not in invalid segments then it is set to current window
            if prev_window == None:
                ind = [i]
                prev_window = window
                continue
                
            if window_type == "dict" and (window["start"] - prev_window["end"]) < consolidation_size:
                prev_window["end"]     =   window["end"]
                prev_window["endLat"]  =   window["endLat"]
                prev_window["endLong"] =   window["endLong"]
                prev_window["endTz"]   =   window["endTz"]
                ind.append(i)
            elif window_type == "arr" and (window[0] - prev_window[1]) < consolidation_size:
                prev_window     =   [prev_window[0], window[1]]    
            else:
                indices.append(copy.deepcopy(ind))
                consolidated.append(prev_window)
                prev_window = window
                ind = [i]

        
        if ind != None:
            consolidated.append(prev_window)
            indices.append(ind)
            
        out = consolidated, indices

    return out


def add_predicted_usage_times(o_df, windows=None, times_probs=None):
    """
    Adds a "predicted_usage" column to the o_df and "usage_probability" column to the o_df
    Args:
        o_df (pd.Dataframe)
        windows (list<dict>): List of dictionaries where each dictionary corresponds to a predicted phone movement
        times_probs (tuple<list<tuple<float>>, list<float>>): Tuple of time periods and predicted phone movement segments
    Returns:
        o_df (pd.Dataframe)
    """
    if len(o_df):
        o_df["usage_probability"] = 0
        o_df["predicted_usage"] = 0
        if not times_probs is None and len(times_probs) > 0:
            for t,p in zip(times_probs[0], times_probs[1]):
                o_df.loc[t[0]:t[1], "usage_probability"] = p
        if not windows is None and len(windows) > 0:
            for w in windows:
                o_df.loc[w["start"]:w["end"], "predicted_usage"] = 1
    return o_df


def reshape_X3(X3, slices_to_use=None):
    """
    Reshapes a 3D (observation x time x sensor) matrix to a 2D matrix 
    Args:
        X3 (ndarray): The 3D matrix
        slices_to_use (list<int>): The indices of the slices to include in the reshaped matrix
    Returns:
        The reshaped matrix
    """
    assert len(X3.shape) == 3
    slices_to_use = np.arange(X3.shape[2]) if slices_to_use is None else slices_to_use
    return np.reshape(X3[:,:,slices_to_use], 
                  (X3[:,:,slices_to_use].shape[0], X3[:,:,slices_to_use].shape[1]*X3[:,:,slices_to_use].shape[2]), 'F')


def process_phone_calls(gps, infraction, usage_windows, speed_cutoff, trip_start, trip_end, 
    usage_cutoff=0.5, signal_cutoff=0.25, check_usage=True):

    """
    Finds intervals for which a phone call was taking place, and applies logic to distinguish 
    hands-free from in-hand calls.

    Phone calls are determined by looking at the infraction data. No infraction data ==> no phone calls.

        Args: 
            gps_df (dict or pd.DataFrame)
            infraction (dict or pd.DataFrame)
            usage_windows (dict): All of the time windows for which phone usage was detected.
            speed_cutoff (float): The speed cutoff for phone usage
            trip_start (float): The trip start time
            trip_end (float): The trip end time
            usage_cutoff (float, 0.5): The percent of moving that indicates phone call in hand
            signal_cutoff (float, 0.25): The amount of bluetooth signal in order to return handsfree phone call
            check_usage: if True, check phone usage when distinguishing hands-free from in-hand calls

        Returns:
            Phone call windows in a similar format to phone usage windows

            valid_call_time: The specified duration of the trip, or the time span of the GPS data frame, 
            whichever is smaller.
    """

    # Initialize the output variables.
    out = []
    valid_call_time = 0

    # Make sure that GPS and infraction data are represented as data frames.
    gps_df = None if gps        is None else pd.DataFrame(gps) 
    inf_df = None if infraction is None else pd.DataFrame(infraction) 
    
    if (validate_dataframe(gps_df, name="gps")):

        gps_df.index = np.float64(gps_df['time_unix_epoch'].values)
        gps_start_time = gps_df['time_unix_epoch'].values[0]
        gps_end_time = gps_df['time_unix_epoch'].values[-1]

        # Set the "valid call time" to be either the full time range of the GPS data, or the
        # total trip duration, whichever is smaller.
        valid_call_time = min(gps_end_time - gps_start_time, trip_end - trip_start)

        # If we have any infraction data, use them to determine the phone call intervals.
        if not inf_df is None and len(inf_df) > 0:
            inf_df = inf_df[inf_df["type"] == "PHONE_CALL"]
            vec = inf_df["value"].values == 1
            
            # JC -- The logic implemented in the next line is causing us to miss phone call detections.
            # The reason is that `vec_to_intervals` marks an "on" interval as beginning at the first "1"
            # instance and ending at the last consecutive "1" instance. But that model doesn't work
            # for PHONE_CALL infractions. PHONE_CALL infractions sometimes have "1" events
            # when the call begins, qnd have "0" events when it ends. Instead we want the full time 
            # between the first "1" and the next most proximal "0". Rather than re-writing that 
            # function (which is called from a few other places), I'm just going to implement the correct 
            # logic here.
            #
            # call_intervals = [(inf_df["time_unix_epoch"].values[I[0]], inf_df["time_unix_epoch"].values[I[1]]) 
            #                   for I in  interval_helpers.vec_to_intervals(vec)]
            
            call_ongoing = False
            call_start_index = None
            infraction_times = inf_df["time_unix_epoch"].values
            call_intervals = []
            for this_index, value in enumerate(vec):

                if call_ongoing:
                    if value == False:  # If this is a "call off" record
                        call_intervals.append((infraction_times[call_start_index], infraction_times[this_index]))
                        call_ongoing = False
                    elif this_index == len(vec) - 1:  # if this is the end of the trip
                        call_intervals.append((infraction_times[call_start_index], trip_end))

                else: # call_ongoing == False
                    if value == True:  # If this is a "call on" record
                        call_start_index = this_index
                        call_ongoing = True

                    else: # value == False
                        if call_start_index is None:  # If there is no currently indicated call start index
                        
                            # Assume that this call was ongoing at the start of the trip. One hack -- 
                            # I'm not confident that all such scenarios represent real call ends. There are
                            # many that occur right when the GPS data begin -- too many to be coincidental.
                            # So here I'm excluding cases that are close to the beginning of the GPS data.
                            if infraction_times[this_index] - gps_start_time > 20.0:

                                # Another hack: some of these cases give us unrealistically long phone calls.
                                # I'm not convinced that these are valid. So here I'm only going to count these
                                # if the time is below some threshold.
                                if infraction_times[this_index] - trip_start < 600.0:
                                    call_intervals.append((trip_start, infraction_times[this_index]))
                            call_ongoing = False

        else:
            call_intervals = []

        # Drop call intervals from the list if they occur entirely within a low speed period.
        if speed_cutoff > 0:
            call_intervals = [c for c in call_intervals if gps_df[c[0]:c[1]]["speed"].max() >= speed_cutoff]

        # Get a list of all the "phone_usage" intervals. These represent times during which the phone 
        # was being actively handled, according to the results of earlier processing.
        usage_intervals  = [(w["start"], w["end"]) for w in usage_windows 
                             if not "method_name" in w or w["method_name"] == "phone_usage"]
            
        # This is a list of values of infraction "audio context" that correspond to the phone being
        # hooked up to some kind of audio thing, like bluetooth ("BT"), the audio jack, or whatever. 
        states = ["BT_A2DP", "BT_HFP", "BT_ACL", "JACK", "USB", "CAR"]

        # This will be a list of boo=lean values indicating whether each call interval is "in hand". 
        call_in_hand = []

        # Loop over  call intervals.
        for i, call_int in enumerate(call_intervals):

            # Get the fraction of this call interval that overlaps some usage interval.
            usage_time  = sum(I[1] - I[0] for I in interval_helpers.intersect_interval_lists([call_int], usage_intervals))
            usage_ratio = 0 if (usage_time == 0) else usage_time / (call_int[1] - call_int[0])

            # If there is enough "activeness" during this call, then label it as active.
            if check_usage and usage_ratio > usage_cutoff:
                in_hand = True

            elif check_usage and usage_ratio < 0.01:
                in_hand = False

            # Otherwise, look at what we know about the audio state. Basically this block says,
            # "If they are using some kind of audio hookup like bluetooth or the phone jack or whatever,
            # then consider this call 'hands-free'".
            else:
                if "audio_context" in gps_df.columns:
                    observed_states = set(gps_df["audio_context"][call_int[0]:call_int[1]].values)
                    cut_df = gps_df[call_int[0]:call_int[1]]
                    if len(cut_df) > 1:
                        cut_len = float(cut_df["time_unix_epoch"].values[-1] - cut_df["time_unix_epoch"].values[0])
                        T = [0 if not cut_len else  
                             time_satisfied(cut_df, lambda df: df["audio_context"].values == s)/cut_len for s in states]
                        if (max(T) > signal_cutoff):
                            in_hand = False
                        else:
                            in_hand = True

                else:
                    in_hand = True

            call_in_hand.append(in_hand)

        # Put together the final list of phone call intervals. Here we distinguish between "in hand"
        # and "hands free" calls using the `is_active` list that we just computed.     
        for a, c in zip(call_in_hand, call_intervals):
            start, end = clamp_interval_to_trip(c, trip_start, trip_end)
            if end > start and (not start is None):
                out.append(create_window(gps_df, start, end, 
                                         method="phone_call_in_hand" if a else "phone_call_hands_free"))

    return out, valid_call_time


def clamp_interval_to_trip(interval, trip_start, trip_end):
    start, end = min(max(interval[0], trip_start), trip_end), min(max(interval[1], trip_start), trip_end)
    if start == end:
        start, end = None, None
    return start, end


def get_screen_state(inf_df, device_type="iOS", include_call=True):
    """
        Args: 
            inf_df (pd.DataFrame): Infraction data frame
            device_type (string): 'iPhone' or 'ANDROID'
            include_call (boolean, default=True): A boolean that specifies whether to count phone calls as screen time

        Returns:
            None if there is no way to know the screen state, and otherwise an interpolater function that maps time to screen
            state
    """
    if inf_df is None or not len(inf_df) > 2:
        return interp1d((0.0, float("inf")), (0.0, 0.0), bounds_error=False, fill_value=0, kind="nearest")
    #Build the screen unlocked trace vector
    if not include_call:
        su_trace = np.logical_and(inf_df["type"] == "SCREEN_UNLOCKED", inf_df["value"] == 1)
    else:
        su_trace = np.logical_and(np.logical_or(inf_df["type"] == "SCREEN_UNLOCKED", inf_df["type"] == "PHONE_CALL"),
                              inf_df["value"] == 1)
    #If Android or Iphone with a passcode
    if (not "ios" in device_type.lower() and not "iphone" in device_type.lower()) or ("passcode" in inf_df and
                                                                                      any(inf_df["passcode"].values == 1)):
        screen_on = su_trace
    else: #If Iphone with no passcode or uncertain passcode
        S = set(inf_df["type"].values)
        if "SCREEN_TURNED_ON" in S or "SCREEN_TURNED_OFF" in S: #Use darwin notifications
            screen_on = np.zeros(len(inf_df))
            T = inf_df["type"].values
            screen_switches = [i for i in xrange(len(T)) if T[i] == "SCREEN_TURNED_ON" or  T[i] == "SCREEN_TURNED_OFF"]
            last = 0
            for ix in screen_switches:
                screen_on[last:ix] = T[ix] == "SCREEN_TURNED_OFF"
                last = ix
            screen_on[last:] = T[last] == "SCREEN_TURNED_ON"
        elif "PHONE_CALL" in S or ("SCREEN_UNLOCKED" in S and all(inf_df["value"].values)):
            #No Darwin and no/unknown passcode Iphone
            return None
        else: #Probably iphone with passcode
            screen_on = su_trace
    return interp1d(inf_df["time_unix_epoch"].values, screen_on, bounds_error=False, fill_value=0, kind="nearest")   


def add_screen_state(o_df, inf_df, device_type="iOS", include_call=True):
    '''
    Adds a screen-on column to the orientation dataframe by using the infraction dataframe.
    '''

    # This call to `get_screen_state` returns an interpolator function that gives screen state
    # as a function of time, or "None" if screen state can't be determined.
    f = get_screen_state(inf_df, device_type=device_type, include_call=include_call)

    # Apply the interpolator if we got one.
    if f is None:
        o_df["screen_on"] = -1
    else:
        o_df["screen_on"] = f(o_df["time_unix_epoch"].values)

    return o_df


def validate_dataframe(df, name=""):
    #This method validates that a dataframe is ready to use
    out = True
    if not isinstance(df, pd.core.frame.DataFrame):
        logger.debug("{} dataframe is not a pandas dataframe".format(name))
        out = False        
    elif df is None:
        logger.debug("{} dataframe is null".format(name))
        out = False
    elif len(df) < 5:
        logger.debug("{} dataframe is less than size 5".format(name))
        out = False
    return out


def compute_passive_windows(orientation_df, gps, active_windows, seg_times, non_active_intervals, use_high_precision, invalidated_segment_times, consolidation_size):
    '''
    Determines windows in which the phone was being used passively. 
    "Passiveness" is determined entirely by the "screen_on" state.

    orientation_df: Orientation data frame.

    gps: GPS data frame.

    active_windows: A list of dictionaries, each describing a time interval during which the
    phone was determined to be "active" -- i.e. being used in some way.

    seg_times: A list of time segments. Each is a tuple: (start, end), expresssed as unix time. 
    This set of time segments spans the entire trip being processed, without any omissions.

    use_high_precision:
    '''

    passive_windows, passive_indices = [], []

    if len(seg_times):

        overall_start_time = seg_times[0][0]
        overall_end_time = seg_times[-1][1]

        # Get a list of the time intervals during which the screen was on.
        screen_on_indices = interval_helpers.vec_to_intervals(orientation_df["screen_on"].values)
        screen_on_intervals = [(orientation_df["time_unix_epoch"].values[ind[0]], 
                                orientation_df["time_unix_epoch"].values[ind[1]]) 
                                for ind in screen_on_indices]

        # Figure out the passive windows.  We are really identifying "passive phone
        # usage" as a subset of *any* phone usage.
        passive_intervals = interval_helpers.intersect_interval_lists(screen_on_intervals, non_active_intervals)
        
        raw_passive_windows = [create_window(gps, t[0], t[1], method="passive_phone_usage",
                                             flags={"activity": 0, "use_high_precision": use_high_precision}) 
                               for t in passive_intervals]
       

        consolidated_passive_windows, consolidated_passive_indices = consolidate_windows(raw_passive_windows, consolidation_size = consolidation_size, invalidated_segment_times = invalidated_segment_times)

        # Sometimes the process of intersecting intervals above leaves us with "passive phone usage" windows
        # that have very short durations and ar probably better considered processing artifacts at the edge
        # of active intervals. So here we look out for such cases. 
        for ii, pw in enumerate(consolidated_passive_windows):
            keep = True
            if pw['end'] - pw['start'] < 10.0:  # If this interval is short...
                for aw in active_windows:
                    # if this passive window is adjacent to some active window...
                    if np.abs(pw['start'] - aw['end']) < 0.01 or np.abs(pw['end'] - aw['start']) < 0.01:
                        keep = False
                        continue
            if keep:
                passive_windows.append(pw)
                passive_indices.append(consolidated_passive_indices[ii])

    return passive_windows, passive_indices


def process_sensors(normalized_sensors, trip_start, trip_end):
    '''
    This method checks and processes the inputs to the predict function.
    Returns a bunch of None values if the input does not contain the correct data.
    '''

    # Handle the orientation dataframe. If it is present and not equal to None, do a couple of
    # bits of processing.
    orientation_df = None
    if 'orientation_df' in normalized_sensors:
        orientation_df = normalized_sensors['orientation_df']
        if orientation_df is not None and len(orientation_df) > 0:

            # Set the index to be the time.
            orientation_df.index = np.float64(orientation_df['time_unix_epoch'].values)

            # Add a column to the orientation data frame that gives the magnitude of the 
            # acceleration vector.
            orientation_df = diff_grav_acc(orientation_df) 

    if orientation_df is not None and len(orientation_df) == 0:
        orientation_df = None

    # Do a similar thing with the GPS data frame.
    gps_df = None
    if 'gps_df' in normalized_sensors:
        gps_df = normalized_sensors['gps_df']
        if gps_df is not None and len(gps_df) > 0:
            gps_df.index = np.float64(gps_df['time_unix_epoch'].values)

    if gps_df is not None and len(gps_df) == 0:
        gps_df = None

    # Set the trip start and end time, if they are not already set. If we have both an orientation data frame
    # and a GPS data frame, then use the overlap time between the two. Otherwise use the time range of one
    # or the other.
    if trip_start is None and trip_end is None:
        if orientation_df is not None and gps_df is not None:
            trip_start = max(orientation_df['time_unix_epoch'].values[0], gps_df['time_unix_epoch'].values[0])
            trip_end = min(orientation_df['time_unix_epoch'].values[-1], gps_df['time_unix_epoch'].values[-1])
        elif gps_df is not None:
            trip_start = gps_df['time_unix_epoch'].values[0]
            trip_end = gps_df['time_unix_epoch'].values[-1]
        elif orientation_df is not None:
            trip_start = orientation_df['time_unix_epoch'].values[0]
            trip_end = orientation_df['time_unix_epoch'].values[-1]

    return orientation_df, gps_df, trip_start, trip_end 


def compute_non_call_usage(gps, usage_windows, phone_call_windows):
    non_call_usage_windows = []
    if usage_windows:
        for kind in ["passive_phone_usage", "phone_usage"]:
            usage_intervals = [(w["start"], w["end"]) for w in usage_windows if w["method_name"] == kind]
            start, end  = min([w["start"] for w in usage_windows]), max([w["end"] for w in usage_windows])
            non_call_intervals = interval_helpers.negate_intervals([(w["start"], w["end"]) for w in phone_call_windows], 
                                                                   start, end)
            non_call_usage_intervals = interval_helpers.intersect_interval_lists(usage_intervals,  non_call_intervals)
            non_call_usage_windows += [create_window(gps, I[0], I[1], method="non_call_{}".format(kind)) 
                                       for I in non_call_usage_intervals]
    return non_call_usage_windows


def determine_device_type(device_type, orientation_df):
    # Returns a string defining a device's type.
    if not device_type is None and ("iphone" in device_type.lower() or "ios" in device_type.lower()):
        out = "iPhone"
    elif not device_type is None:
        out = "ANDROID"
    else:
        out = "iPhone" if "user-accel_x" in orientation_df or "user_accel_x"  in orientation_df else "ANDROID"
    return out


def get_valid_usage_time(times, validated_sq, active_windows):
    '''
    times: A list of time intervals. Typically this consists of all of the 10-second intervals for the trip.

    validated_sq: Boolean flags indicating whether the intervals pass a speed / quality check.

    active_windows: The time windows considered "valid". Typically these will be consolidated windows -- i.e. 
    ones that have merged multiple nearby "active" segments. That is, there might be "invalid" segments
    in the "times" list that fall inside some active window, because of the consolidation process. If such an 
    "invalid" segment falls into an "active" window, it is counted towards the total "valid_usage_time" anyway.

    In other words, this tallys up all "times" that either (1) passed the speed and quality tests, 
    or (2) were lumped in with an active window due to the consolidation process.
    '''

    # Separate the list of input time intervals into separate "valid" and "invalid" subsets.
    # "Invalid" means having low speed or poor quality.
    valid_times, invalid_times = [],[]
    for i, t in enumerate(times):
        if validated_sq[i]:
            valid_times.append(t)
        else:
            invalid_times.append(t)

    included_invalid_times = interval_helpers.intersect_interval_lists(invalid_times, 
                                                                       [(w["start"], w["end"]) for w in active_windows])
    valid_times = interval_helpers.union_interval_list(included_invalid_times + valid_times)
    return sum([t[1] - t[0] for t in valid_times]), valid_times


    
# NOTICE: THIS PROGRAM CONSISTS OF TRADE SECRETS THAT ARE THE PROPERTY OF TRUEMOTION, INC. AND ARE PROPRIETARY MATERIAL OF TRUEMOTION, INC.