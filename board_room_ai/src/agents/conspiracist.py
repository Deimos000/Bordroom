class Conspiracist:
    def __init__(self):
        pass

    def analyze_insider_activity(self, insider_buys, insider_sells):
        """
        Logic: Simple heuristic. If Insider_Sells > Insider_Buys by 20%, Output = Negative Signal.
        """
        if insider_buys == 0:
            if insider_sells > 0:
                return -1.0 # Only selling
            else:
                return 0.0 # No activity

        ratio = (insider_sells - insider_buys) / insider_buys

        if ratio > 0.20:
            return -1.0 # Negative signal
        elif ratio < -0.20: # Sells < Buys significantly -> Buys > Sells
            return 1.0 # Positive signal
        else:
            return 0.0
