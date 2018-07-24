def __match_gc_with_afp(departure, 
                        Trajectory, 
                        Traj_KDtree, 
                        afp):
    """
    Trajectory should be a LineString
    TrajID should be the FID of the matching Trajectory (center)
    afp should be a 1-D np array containing 
    [HEADID, GEOM, alt_min, alt_max, ST, ET, DURATION, avg delay, max delay, avg rate]
    Traj_KDtree is a tree based on traj_coords
    [kdtree, elap_time, alt (in feet)]
    
    Return:
    [HEADID, duration, avg rate, avg delay, max delay]
    """

    # intersection returns the line segment within the polygon
    try:
        geo_type = afp[1].geom_type
    except AttributeError:
        geo_type = type(afp[1])
        
    if geo_type == np.float:
        return [-1, -1, -1, -1, -1]
    elif geo_type == 'Polygon':
        interLine = Trajectory.intersection(afp[1])
        try:
            EntryPt = interLine.coords[0][:2] # entry point
            ExitPt = interLine.coords[-1][:2] # exit point
        except NotImplementedError:
            interLine = list(interLine)
            try:
                EntryPt = interLine[0].coords[0][:2] # entry point
                ExitPt = interLine[-1].coords[-1][:2] # exit point
            except IndexError:
#                 print(departure, ' no geo match')
                return [-1, -1, -1, -1, -1]
            
        CrossPt_Combine = np.array((EntryPt, ExitPt))
        # find two closest points (index) for entry/ exit point
        nearestidx = Traj_KDtree[0].query(CrossPt_Combine, k = 2)[1]

        # Use the average of the time of the two closest points for entry/exit point as the altitude
        Entry_Altitude = np.mean(Traj_KDtree[2][nearestidx[0]])
        Exit_Altitude = np.mean(Traj_KDtree[2][nearestidx[1]])
            
        if (Entry_Altitude>=afp[2]*100 and Entry_Altitude<=afp[3]*100) or (Exit_Altitude>=afp[2]*100 and Exit_Altitude<=afp[3]*100):

            # print("altitude is ok %d"%saa[0])
            # Use the average of the time of the two closest points for entry point as the crossing time
            Entry_DeltaSec = np.mean(Traj_KDtree[1][nearestidx[0]])
            Exit_DeltaSec = np.mean(Traj_KDtree[1][nearestidx[1]])

            EntryTime = departure + timedelta(seconds = Entry_DeltaSec)
            ExitTime = departure + timedelta(seconds = Exit_DeltaSec)

            if EntryTime <= afp[5] and ExitTime >= afp[4]:
                # Entry time should be earlier than ALERT STOP TIME and exit time should be later than ALERT START TIME
                TraverseTime = abs(Exit_DeltaSec - Entry_DeltaSec)
                if TraverseTime <= 0:
#                     print(departure, ' negative traverse time')
                    return [-1, -1, -1, -1, -1]
                else:
                    return [afp[0], afp[6], afp[7], afp[8], afp[9]]
            else:
#                 print(departure, ' crossing time out of afp time')
                return [-1, -1, -1, -1, -1]
        else:
#             print(departure, ' altitude restriction not satisfied')
            return [-1, -1, -1, -1, -1]
        
    elif geo_type == 'LineString':
        interPt = Trajectory.intersection(afp[1])
        try:
            interPt_coord = list(interPt.coords[0])
        except NotImplementedError:
            
            interPt = list(interPt)
            if len(interPt) == 0:
                return [-1, -1, -1, -1, -1]
            else:
                interPt_coord = [list(tmp_pt.coords[0]) for tmp_pt in interPt]
        interPt_coord = np.array(interPt_coord).reshape(-1, 2)
        # find two closest points (index) for entry/ exit point
        nearestidx = Traj_KDtree[0].query(interPt_coord, k = 2)[1]

        # Use the average of the time of the two closest points for all points as the altitude
        Altitude_list = [np.mean(Traj_KDtree[2][nearestidx[i]]) for i in range(nearestidx.shape[0])]

        if any((__alt >= afp[2] * 100 and __alt <= afp[3] * 100) for __alt in Altitude_list):
            crosspt_time_list = [(departure + datetime.timedelta(seconds = np.mean(Traj_KDtree[1][nearestidx[i]]))) for i in range(nearestidx.shape[0])]

            if any((__cp_time <= afp[5] and __cp_time >= afp[4]) for __cp_time in crosspt_time_list):
                return [afp[0], afp[6], afp[7], afp[8], afp[9]]
            else:
                return [-1, -1, -1, -1, -1]
        else:
            return [-1, -1, -1, -1, -1]
    else:
        print(geo_type)
        raise ValueError('Type not understood')
        
def match_gc_with_afp(gc_traj, 
                      flight_info, 
                      afp_data):

    Airborne = gc_traj[-1, 3]
    st = time.time()
    # construct linstring and kdtree
    Traj_Line = LineString(coordinates = gc_traj[:, :2])
    Traj_Tree = [KDTree(gc_traj[:, :2]), gc_traj[:, 3], gc_traj[:, 2] * 100]
    
    Mapping_result = {}
    for i in range(flight_info.shape[0]):
        if i % 200 == 0:
            print(i, time.time() - st)
            
        FFID = flight_info[i, 0]
        departureTime = flight_info[i, 1]
        
        EndTime = departureTime + datetime.timedelta(seconds = Airborne)
        ValidAFP = afp_data.loc[(afp_data.ETASTART < EndTime) & (afp_data.ETASTOP > departureTime), 
                                ['HEADID', 'geometry','FCA_FLOOR', 'FCA_CEIL', 'ETASTART', 'ETASTOP', 
                                 'ETA_DUR', 'DLY_AVG', 'MAX_DLY', 'AAR_AVG']]
        
        if ValidAFP.shape[0] == 0:
            Mapping_result[FFID] = np.array([[-1, -1, -1, -1, -1]])
        else:
            Mapping_result[FFID] = []
            for idx, afp in enumerate(ValidAFP.values):                        
                Mapping_result[FFID].append(__match_gc_with_afp(departureTime, 
                                                                Traj_Line, 
                                                                Traj_Tree, 
                                                                afp))
            Mapping_result[FFID] = np.unique(np.array(Mapping_result[FFID]), axis=0).reshape(-1, 5)
    return Mapping_result

def Count_Max_AFP(Mapping_result):
    max_afp = 0
    k = 0 # nonzero SAA traj.
    for FFID in Mapping_result.keys():
        count_afp = np.count_nonzero(Mapping_result[FFID][:,0] != -1)
        if count_afp != 0:
            k += 1
        if count_afp > max_afp:
            max_afp = count_afp
        else:
            pass
    return max_afp, k