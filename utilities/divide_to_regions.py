import numpy as np


class EEGRegionsDivider:
    def __init__(self):
        # Definition of electrodes for each EEG region according to the 10-20 system
        self.fp = np.sort(np.array([27, 33, 34, 38, 39, 47, 48, 26, 20, 19, 12, 11, 3, 2, 222]))
        self.f = np.sort(np.array([16, 22, 23, 24, 28, 29, 30, 35, 36, 40, 41, 42, 49, 50, 21, 15, 7, 14, 6,
                                   207, 13, 5, 215, 4, 224, 223, 214, 206, 213, 205]))
        self.c = np.sort(np.array([9, 17, 43, 44, 45, 51, 52, 53, 57, 58, 59, 60, 64, 65, 66, 71, 72, 8,
                                    81, 186, 198, 197, 185, 132, 196, 184, 144, 204, 195, 183, 155,
                                   194, 182, 164, 181, 173])) #E257 removed
        self.t = np.sort(np.array([55, 56, 62, 63, 69, 70, 74, 75, 84, 85, 96, 221, 212, 211, 203,
                                   202, 193, 192, 180, 179, 171, 170]))
        self.p = np.sort(np.array([76, 77, 78, 79, 80, 86, 87, 88, 89, 97, 98, 99, 100, 110, 90,
                                   101, 119, 172, 163, 154, 143, 131, 162, 153, 142, 130, 161,
                                   152, 141, 129, 128]))
        self.o = np.sort(np.array([107, 108, 109, 116, 117, 118, 125, 126, 160, 151, 140, 150,
                                   139, 127, 138]))

    def get_all_regions(self):
        """Returns the regions with their respective channels in a single string for each region."""
        return [
            f"1_Fp_chs = {self._format_channels(self.fp)}",
            f"2_F_chs = {self._format_channels(self.f)}",
            f"3_C_chs = {self._format_channels(self.c[:-1])}",
            #f"3_C_chs = {self._format_channels(self.c[:-1])}, Vertex Reference",  # Adds "Vertex Reference"
            f"4_T_chs = {self._format_channels(self.t)}",
            f"5_P_chs = {self._format_channels(self.p)}",
            f"6_O_chs = {self._format_channels(self.o)}"
        ]

    def get_index_channels(self):
        """Returns all channel indices sorted."""
        return np.sort(np.concatenate([self.fp, self.f, self.c, self.t, self.p, self.o]))

    def _format_channels(self, region_array):
        """Private method to format electrodes as strings, prefixed with the letter 'E'."""
        return ', '.join(['E' + str(ch) for ch in region_array])

