from datetime import datetime
import pytz


def currenttime():
    # Get local time to name files
    mytz = pytz.timezone("America/Ensenada")
    mytime = datetime.now(mytz)
    filename = "hydrophone_recordings/" + mytime.strftime("%Y%m%d_%H%M%S") + ".wav"
    return mytime.strftime("%Y%m%d_%H%M%S")
