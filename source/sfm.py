

class SFM:
    def __init__(self, views, matches):
        self.views = views
        self.matches = matches
    
    def compute_pose(self, view1, view2=None, isBaseLine=False):
        if isBaseLine and view2:
            matchObject = self.matches[(view1.name, view2.name)]
            baselinePose = Baseline(view1, view2, matchObject)

    def reconstruct (self):
        baselineView1 = self.views[0]
        baselineView2 = self.views[1]
        self.computePose (view1=baselineView1, view2=baselineView2, isBaseLine=True)
