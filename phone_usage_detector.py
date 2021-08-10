# NOTICE: THIS PROGRAM CONSISTS OF TRADE SECRETS THAT ARE THE PROPERTY OF TRUEMOTION, INC. AND ARE PROPRIETARY MATERIAL OF TRUEMOTION, INC.
__author__ = 'dshieble'

"""
    This is the Phone Usage Detector that determines during which portions of a trip a user held
    his/her phone in hand
"""

import os
import msgpack
import logging
import cPickle
import time
from copy import deepcopy
import gzip

import pandas as pd
import numpy as np

from Minsky.algorithms.interface import SingleModel
from Events.helpers.validate import validate_events
from Events.helpers import phone_usage_helpers
from Events.helpers import interval_helpers
from Events.helpers import sample_generation
from Events.algorithms.event_predictor import EventPredictorFinal

logger = logging.getLogger('DI')

class PhoneUsageDetector(SingleModel):
    __METHOD_NAME__ = "phone_usage"
    __WANTS__ = ["normalized_sensors", "infraction", "device_type", "trip_start", "trip_end"]

    def __init__(self, column_store, cache=True, **kwargs):
        super(PhoneUsageDetector, self).__init__(column_store, cache=True, **kwargs)

    def default_prediction(self):
        return {"windows":[], "legal_usage_periods":[], "valid_usage_time":0, "valid_call_time":0}

    def validate_inputs(self, **kwargs):
        flag1 = (
            kwargs.get('normalized_sensors') is not None and \
            kwargs['normalized_sensors'].get('orientation_df') is not None and \
            len(kwargs['normalized_sensors']['orientation_df'])>0
        )

        flag2 = (
            kwargs.get('normalized_sensors') is not None and \
            kwargs['normalized_sensors'].get('gps_df') is not None and \
            len(kwargs['normalized_sensors']['gps_df'])>0
        )
        return flag1 and flag2

    #Loads in the parameters
    def create_model(self):
        """
        The keys of the saved model are:
            mount_prob_cutoff: The cutoff certainty for the mount detector
            grav_angle_thresholds (float): The cutoff grav-diff-angle
            window_size (float): The length of the window we are looking at (in seconds)
            check_up (boolean): Whether to throw out windows where the phone does not face up
            consolidation_size (int): The length of time between 2 windows where the windows should be consolidated into one
            offset (float): The number of seconds after the phone is picked up to check if the phone is in use
            start_offset (float): The number of seconds after the start of the trip to start scanning for usage
            end_offset (float): The number of seconds before the end of the trip to stop scanning for usage
            speed_cutoff (float): The cutoff speed for determining a period as a usage period
            max_dropout (float): The maximum amount of sensor dropout to predict over an interval
            min_length (int): Minimum length of a usage window
            filter_on_screen (boolean): Whether to filter the usage periods on the screen state
            precision_level (string): low, high or auto
            prob_cutoffs (array-like <float>): High and Low Precision probability cutoffs
            mount_prob_cutoffs (array-like <float>): High and Low Precision probability cutoffs for the mount
            clf (sklearn-like classifier object): Model-based classifier
        """
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            'default_models',
                            'phone_usage_params.msgpack.gz')

        with gzip.open(path,'rb') as f:
            params = msgpack.loads(f.read())
        self.initialize_params(params)
        return params


    def initialize_params(self, params):
        self.grav_angle_thresholds    = params["grav_angle_thresholds"]
        self.window_size              = params["window_size"]
        self.check_up                 = params["check_up"]
        self.consolidation_size       = params["consolidation_size"]
        self.start_offset             = params["start_offset"]
        self.end_offset               = params["end_offset"]
        self.speed_cutoff             = params["speed_cutoff"]
        self.max_dropout              = params["max_dropout"]
        self.min_length               = params["min_length"]
        self.filter_on_screen         = params["filter_on_screen"]
        self.precision_level          = params["precision_level"]
        self.prob_cutoffs             = params["prob_cutoffs"]
        self.min_usage_time           = params["min_usage_time"]
        self.clf                      = cPickle.loads(params["clf"])
        self.params = params

    @validate_events
    def predict(self, model, normalized_sensors, infraction, device_type, trip_start, trip_end,
                reinitialize = True, modify=False, debug=False, postprocess=True):
        """
        This method finds the windows of time where the user is interacting with his/her phone by using motion sensors
        and screen infractions.

        Args:
            model (dict): A dictionary of parameters. Also includes a trained phone use detector.
            normalized_sensors (dict of 'gps_df' and 'orientation_df')
            infraction (pd.DataFrame)
            device_type (string)
            trip_start (float)
            trip_end (float)
            reinitialize (bool, default=True): Whether to recompute the model parameters
            modify (bool, default=False): Whether to modify the normalized_sensors

         Returns:
            List of dictionaries where each dictionary corresponds to a phone movement event and contains the fields
            'method_name', 'start', 'end', 'startLat', 'endLat', 'startLong', 'endLong', 'startTz', 'endTz
        """

        # This is a dictionary into which we will pack various types of debug info that will be used later.
        dbdata = {}

        # Save the parameter values that we're using.
        if debug:
            dbdata['params'] = self.params

        if reinitialize:
            self.initialize_params(model)

        # The "modify" flag controls whether the input sensor data frames get augmented to include some
        # new information that will be calculate below. This is mostly for debugging. If we *don't* want
        # that info returned, then make copies of the relevant data frames exclusively for local use.
        if not modify:
            normalized_sensors = deepcopy(normalized_sensors)
            infraction = None if infraction is None else deepcopy(infraction)

        # Prepare the infraction data frame.
        if infraction is None or len(infraction) == 0:
            infraction = pd.DataFrame([])
        else:
            infraction.index = infraction['time_unix_epoch']

        # Do some preprocessing of the sensor data.
        orientation_df, gps_df, trip_start, trip_end = phone_usage_helpers.process_sensors(normalized_sensors,
                                                                                        trip_start, trip_end)

        # As normalized_sensors has an additional 5 mins past trip_end added to end
        # we must ensure this is removed when calcualting phone usage 
        if trip_end != None:
            
            if orientation_df is not None:
                orientation_df = orientation_df[orientation_df.time_unix_epoch < trip_end]
                
            if gps_df is not None:
                gps_df = gps_df[gps_df.time_unix_epoch < trip_end]

        # Initialize the output to a default that essentially says that there was no phone usage.
        out = self.default_prediction()
        seg_data, seg_times, seg_probs, active_windows, passive_windows, usage_windows = [], [], [], [], [], []

        # If we have orientation data, then enter the next big block, which computes phone usage.
        # Either way, set a flag indicating whether phone usage data are valid.
        if orientation_df is None:
            usage_is_valid = False
        else:
            usage_is_valid = True

            # Add screen info to the orientation df. The new column will be named "screen_on".
            device_type = phone_usage_helpers.determine_device_type(device_type, orientation_df)
            orientation_df = phone_usage_helpers.add_screen_state(orientation_df, infraction,
                                                                  device_type=device_type, include_call=True)
            if debug:
                dbdata['orientation_df'] = deepcopy(orientation_df)

            # Set a flag that indicates whether to use a "high-precision" procedures.  This mostly
            # affects the choice of various thresholds used below.
            use_high_precision, trust_screen   = self.set_high_precision(orientation_df, device_type, infraction)

            # Segment the time series. That is, break up the data into chunks of a certain duration.
            # This call to `make_segments` returns `segments`: time-interpolated
            # data for each time segment, `seg_times`: segment start and end times, `validated_sq`: boolean
            # values indicating whether each segment passes speed and quality tests, and `validated_us`:
            # boolean values indicating whether each segment passes phone usage tests. For the latter two,
            # True values indicate that the segment should be considered for possible distration events.
            seg_data, seg_times, validated_sq, validated_us =  self.make_segments(orientation_df, gps_df, trip_start,
                                                                              trip_end, use_high_precision)
            if debug:
                dbdata['seg_data'] = deepcopy(seg_data)
                dbdata['seg_times'] = deepcopy(seg_times)
                dbdata['validated_sq'] = deepcopy(validated_sq)
                dbdata['validated_us'] = deepcopy(validated_us)

            # The following block gets the "usage windows" for this trip. Usage windows are of two types:
            # *any* phone usage, and "passive" phone usage. These time windows are labeled "phone_usage"
            # and "passive_phone_usage" respectively.
            # The condition says to do this if (1) we have orientation data
            if (not self.filter_on_screen or len(infraction)) and (not orientation_df is None) and len(seg_times):

                # Get the model-based probability of phone usage in all of the time segments.
                # These values are based on a trained classifier.
                seg_probs = self.get_phone_usage_probabilities(seg_data)
                if debug:
                    dbdata['seg_probs'] = deepcopy(seg_probs)

                # Get the usage times and probabilities from the segments. Mostly this filters out segments
                # in which the phone looks like it was "down", or the vehicle speed was below threshold.
                # The probabilities returned here are just
                # copies of the probabilities that were passed in, for the "up" segments only.
                usage_seg_times, usage_seg_probs, non_usage_seg_times, invalidated_segment_times = self.identify_usage_segments(seg_times, seg_probs,
                                                                        np.logical_and(validated_sq, validated_us),
                                                                        orientation_df, gps_df, use_high_precision,
                                                                        device_type)
                if debug:
                    dbdata['usage_seg_times'] = deepcopy(usage_seg_times)
                    dbdata['usage_seg_probs'] = deepcopy(usage_seg_probs)


                # Predict over the segments and create usage windows based on the small prob cutoff. Basically
                # `usage_seg_activity_levels` is a re-scaling of the usage probabilities such that 0 represents something
                # that is at or below the cutoff threshold, and higher numbers represent greater exceedance
                # of that threshold.
                bottom = self.prob_cutoffs[use_high_precision][0]
                usage_seg_activity_levels = [max(0, (p-bottom)/(1-bottom)) for p in usage_seg_probs]
                if debug:
                    dbdata['usage_seg_activity_levels'] = deepcopy(usage_seg_activity_levels)

                # Define the "phone usage" windows. That is, this defines a bunch of windows in which *any* phone usage
                # has been detected. These windows will be flagged with the value "phone_usage" in a field named
                # `method_name`.
                raw_active_windows = [phone_usage_helpers.create_window(gps_df, t[0], t[1],
                                      flags={"activity":a, "use_high_precision":use_high_precision}, method="phone_usage")
                                      for t, a in zip(usage_seg_times, usage_seg_activity_levels)]
                if debug:
                    dbdata['raw_active_windows'] = deepcopy(raw_active_windows)
                    
                passive_window_times = [window['start'] for window in non_usage_seg_times]
                    
                # Commbine windows that are closer together than some threshold.
                active_windows, active_indices = phone_usage_helpers.consolidate_windows(raw_active_windows,
                                                                                         consolidation_size = self.consolidation_size,  invalidated_segment_times = invalidated_segment_times + passive_window_times)
                if debug:
                    dbdata['active_windows'] = deepcopy(active_windows)
                    dbdata['active_indices'] = deepcopy(active_indices)

                # Filter on a minimum amount of usage to ding the user for.
                if len(active_windows) and sum(w["end"] - w["start"] for w in active_windows) < self.min_usage_time:
                    active_windows, active_indices = [], []

                for inds, w in zip(active_indices, active_windows):
                    w["activity"] = [raw_active_windows[i]["activity"] for i in inds]

                # Compute passive usage windows, if we trust the screen state.
                # The `compute_passive_windows` routine defines "passive" as having the screen turned off.
                if trust_screen:
                    passive_windows, _ = phone_usage_helpers.compute_passive_windows(orientation_df, gps_df, 
                                                                                     active_windows, seg_times,
                                                                                     non_usage_seg_times,
                                                                                     use_high_precision,
                                                                                     invalidated_segment_times,
                                                                                    2)
                if debug:
                    dbdata['passive_windows'] = deepcopy(passive_windows)

                # Combine the "active" windows with the "passive" ones. Note that "active" is a misnomer --
                # what the code calls "active" is really just *any* phone usage.
                usage_windows = sorted(active_windows + passive_windows, key=lambda w: w["start"])

            # Figure out the "valid usage time". Note that depending on whether orientation data are available,
            # the "active_windows" parameter might be empty at this point. That's OK. But more typically it will
            # contain a non-empty set of consolidated "active windows" -- i.e. the ones with a type of "phone_usage".
            out["valid_usage_time"], out["legal_usage_periods"] = phone_usage_helpers.get_valid_usage_time(
                seg_times, validated_sq, active_windows)

            if modify and len(seg_times):
                normalized_sensors["orientation_df"] = phone_usage_helpers.add_predicted_usage_times(orientation_df, active_windows,
                    (seg_times, seg_probs))
                
        # end of "if not orientation_df is None:" That is, that block was what we do if we have an orientation data frame.

        # Determine time windows for which a phone call was taking place. The intervals in the returned `phone_call_windows`
        # list will be labeled either "phone_call_in_hand" or "phone_call_hands_free".
        phone_call_windows, out["valid_call_time"] = phone_usage_helpers.process_phone_calls(gps_df, infraction, usage_windows,
                                                                                      self.speed_cutoff, trip_start, trip_end,
                                                                                      check_usage=usage_is_valid)

        non_call_usage_windows = phone_usage_helpers.compute_non_call_usage(gps_df, usage_windows, phone_call_windows)

        # Put together all of the various types of windows and return the result.
        out["windows"] = sorted(usage_windows + phone_call_windows + non_call_usage_windows, key=lambda w: w["start"])

        # Save data for debugging.
        if debug:
            out['dbdata'] = dbdata

        if postprocess:
            out = EventPredictorFinal.postprocess(out)
        return out


    def identify_usage_segments(self, seg_times, seg_probs, validated, orientation_df, gps_df, use_high_precision, device_type):
        """
            Identifies time segments containing phone usage by combining model-based estimates with a rule-based model.

            Args:
                seg_times (array<tuple<float>>): Start and end times for a set of time segments.
                seg_probs (array<float>): Array of usage probabilities for each time segment, from a trained classifier.
                validated (array<boolean>): Array storing whether each interval is valid for phone usage
                orientation_df (pd.DataFrame)
                gps_df (pd.DataFrame)
                use_high_precision (boolean): Whether to use high precision
                device_type (string)

            Returns:
                Arrays storing the time windows of phone usage and their corresponding probabilities.
        """

        # Initialize the arrays to be returned.
        usage_seg_times = []
        usage_seg_probs = []

        # This indicates valid times in which the phone was not in use
        non_usage_seg_times = []
        
        invalidated_segment_times = []


         # Get the maximum grav diff angle for each period.
        seg_max_gda = [self.max_gda(orientation_df, t[0], t[1]) for t in seg_times]

        if len(seg_times) and not (use_high_precision and not (max(seg_probs) >= self.prob_cutoffs[use_high_precision][2]
                                     or max(seg_max_gda) >= self.grav_angle_thresholds[use_high_precision])):

            # Start by assuming that the state is "down".
            status = "down"

            # Loop over the list of segments.
            for i in xrange(len(seg_times)):

                # If this segment did not pass the validation checks (which are summarized by the `validated`
                # input list), then mark the phone state as being "down".
                if not validated[i]:
                    invalidated_segment_times.append(seg_times[i])
                    status = "down"
                    continue

                # Condition for changing to up state if currently in down state.
                down_to_up = ((seg_max_gda[i] >= self.grav_angle_thresholds[use_high_precision]) and
                                seg_probs[i] >= self.prob_cutoffs[use_high_precision][1])

                # Condition for staying in up state if currently in up state.
                still_up = seg_probs[i] >= self.prob_cutoffs[use_high_precision][0]

                if (status == "down" and down_to_up) or (status == "up"  and still_up):
                    status = "up"
                    usage_seg_times.append(seg_times[i])
                    usage_seg_probs.append(seg_probs[i])
                else:
                    status = "down"
                    non_usage_seg_times.append(seg_times[i])

        return usage_seg_times, usage_seg_probs, non_usage_seg_times, invalidated_segment_times


    def get_phone_usage_probabilities(self, seg_data):
        """
        This method runs a model to estimate probability of phone usage for each time segment.

        Args:
            seg_data (list): Sensor data associated with some number of time segments.

        Returns:
            A list of usage probabilities corresponding to each element of `seg_data`,
            as determined by a previously trained model.
        """

        # Note "clf" is a trained `scikit` classifier. "predict_proba" is a method that gives the probability
        # of class membership; "class" presumably being "in use".
        return [] if not len(seg_data) else self.clf.predict_proba(np.vstack(seg_data))[:, 1]


    def make_segments(self, o_df, gps, start_time, end_time, use_high_precision):
        """
        This method splits the trip into short windows.

        Args:
            o_df (pd.DataFrame)
            gps (pd.DataFrame)
            start_time (float)
            end_time (float)
            use_high_precision (boolean)

        Returns:
            Information about time segments and validity.
        """

        # Get the segments. The returned values are `segments`: the time-interpolated data for each
        # segment / window, and `times`, which gives the segment start and end times.
        seg_data, seg_times = sample_generation.generate_segments(o_df, start_time, end_time,
                                                              window_size=self.window_size)

        # Return empty arrays if we got no time segments.
        if len(seg_data) == 0:
            out = [], [], [], []

        # Otherwise apply some filters to remove some of the segments from consideration.
        else:
            # Note the following two lines produce arrays of booleans indicating whether each
            # segment is "valid" -- i.e. is something we want to examine for distraction events.
            # Note this is done by looking back at the original orientation data frame,
            # not from looking at the interpolated data in `segments`.
            validated_sq = [self.filter_speed_and_quality(t, gps, o_df, start_time, end_time) for t in seg_times]
            validated_us = [self.filter_on_usage(t, o_df) for t in seg_times]
            out =  seg_data, seg_times, validated_sq, validated_us

        return out


    def filter_speed_and_quality(self, time_period, gps_df, orientation_df, start_time, end_time):
        """
        This method determines if a segment is valid for inclusion based on the sensors and speed.

        Args:
            time_period (tuple<float>): (start time of period, end time of period)
            gps_df (pd.DataFrame)
            orientation_df (pd.DataFrame)
            start_time (float): start of trip
            end_time (float): end of trip

        Returns:
            True or False
        """
        o_slice = orientation_df[time_period[0]:time_period[1]]
        if time_period[1] - time_period[0] < self.min_length:
            return False

        # If some "max_dropout" vlaue is specified...
        if not self.max_dropout is None:
            # If we have fewer than 2 records, take that as an indication of having insufficient data
            # and return False.
            if len(o_slice["time_unix_epoch"].values) <= 2:
                return False
            # Otherwise, check the time gaps in the data. (This is what is meant by "dropout":
            # a time gap between consecutive data points.) If any exceeds the dropout threshold,
            # retiurn False.
            elif max(np.diff(o_slice["time_unix_epoch"].values)) > self.max_dropout:
                return False

        # Filter on speed. Return False if all GPS records in the interval of interest have a speed
        # below a given threshold.
        if not self.speed_cutoff is None:
            gps_start_ind = min(len(gps_df) - 1, gps_df["time_unix_epoch"].searchsorted(time_period[0])[0])
            gps_end_ind   = min(len(gps_df) - 1, gps_df["time_unix_epoch"].searchsorted(time_period[1])[0])
            if (gps_end_ind - gps_start_ind >= 2) and (
                np.max(gps_df["speed"].values[gps_start_ind:gps_end_ind]) < self.speed_cutoff):
                return False

        # Return False if this time interval is too close to the trip start or end.
        if self.end_offset or self.start_offset:
            if (time_period[0] < start_time + self.start_offset) or (time_period[1] + self.end_offset > end_time):
                return False

        # If we passed all the tests abiove, return True.
        return True


    def filter_on_usage(self, time_period, orientation_df):

        """
        This method determines if a segment is valid for inclusion based on the screen state.

        Args:
            time_period (tuple<float>): (start time of period, end time of period)
            orientation_df (pd.DataFrame): Orientation data frame.

        Returns:
            True or False
        """
        orientation_slice = orientation_df[time_period[0]:time_period[1]]

        # Return False if the screen is never on during the time period. (Only if we are checking
        # screen state in the first place.)
        if self.filter_on_screen:
            if not (len(orientation_slice) and any(orientation_slice["screen_on"].values)):
                return False

        # Return False if the screen was face down over the entire time period.
        # Here we define screen face down as having an angle > 45 degrees
        if self.check_up and np.min(orientation_slice["grav_z"]) > 5:
            return False

        # If we got through that, return True.
        return True


    def max_gda(self, o_df, start, end):
        # Returns the maximum GDA during the window
        GDA = o_df[start:end]["gravity_angle_diff"].values
        return -1 if not len(GDA) else max(GDA)


    def set_high_precision(self, orientation_df, device_type, inf_df):
        # Determines the precision level based on the model settings, the device type and whether the user has a passcode.

        # Trust the screen if and only if there are no "-1" values in the "screen_on" column of the
        # orientation data frame. "-1"s basically indicate unknown screen state.
        trust_screen = not -1 in orientation_df["screen_on"].values

        # If the precision level is already set, use it.
        if self.precision_level.lower() == "low":
            high_precision = False
        elif self.precision_level.lower() == "high":
            high_precision = True
        else:
            # Turn high precision on if we don't trust the screen, or if we trust the screen but there is no passcode
            # This means that we use the high precision classifier for android no passcode trips
            high_precision = (not trust_screen) or ("passcode" in inf_df and all(inf_df["passcode"].values == 0))

        return high_precision, trust_screen


# NOTICE: THIS PROGRAM CONSISTS OF TRADE SECRETS THAT ARE THE PROPERTY OF TRUEMOTION, INC. AND ARE PROPRIETARY MATERIAL OF TRUEMOTION, INC.
