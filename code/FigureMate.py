"""
Author: Omar M Khalil 170018405

A class to create FigureMate objects, which, are used to store
    - x and y axis label strings,
    - title string, and
    - save path string.
"""

class FigureMate:
    """
    A class to create objects which store strings containing the title, x-axis label, y-axis label, and save path strings for a figure.
    """

    def __init__(self, x_lbl='', y_lbl='', heading='', plot_type='', path=''):
        """
        init method - constructs a FigureMate object.
        Parameters:
            x_lbl:      x axis label - pass None or '' if x label not required.
            y_lbl:      y axis label - pass None or ''  if y label not required.
            heading:    Figure heading string - passing '' will cause a heading
                        to be auto-generated. Passing None will prevent a title
                        from being rendered.
            plot_type:  visualisation type or semantics. Used by the object in
                        cases were no heading is provided to auto construct a
                        heading.
            path:     save path string. If empty string given (default), the
                        object will auto generate a figure save label as long
                        as at least an x or y label is provided. Otherwise, the
                        figure will not be saved in compatible plot methods.
        """
        self.setup_params(x_lbl, y_lbl, heading, plot_type, path)


    def get_x(self):
        return self.x_lbl

    def get_y(self):
        return self.y_lbl

    def get_plot_type(self):
        return self.plot_type

    def get_heading(self):
        return self.heading

    def get_path(self):
        return self.path

    def set_x(self, x_lbl):
        self.x_lbl = x_lbl
        self.setup_params(x_lbl, y_lbl, heading, plot_type, path)

    def set_y(self, y_lbl):
        self.y_lbl = y_lbl
        self.setup_params(x_lbl, y_lbl, heading, plot_type, path)

    def set_plot_type(self, plot_type):
        self.plot_type = plot_type
        self.setup_params(x_lbl, y_lbl, heading, plot_type, path)

    def set_heading(self, heading):
        self.heading = heading

    def set_path(self, path):
        self.path = path

    def setup_params(self, x_lbl, y_lbl, heading, plot_type, path):
        self.x_lbl = x_lbl
        self.y_lbl = y_lbl
        heading_affix = " plot of " + y_lbl + " vs " +  x_lbl
        if heading != '' and heading != None:
            # heading given
            self.heading = heading
        elif heading == None:
            # no heading wanted
            self.heading = ''
        elif heading == '':
            # heading not given - auto reconstruct
            if plot_type != '':
                # plot_type given
                self.plot_type = plot_type
                if x_lbl != '' and y_lbl != '':
                    # both x and y labels given
                    self.heading = plot_type + heading_affix
                elif x_lbl != '':
                    # x label given but not y
                    self.y_lbl = plot_type
                    self.heading = plot_type + heading_affix
                else:
                    # y label given but not x, or neither x nor y given
                    self.heading = plot_type + " plot"
            else:
                # plot_type not given
                if x_lbl != '' and y_lbl != '':
                    # both x and y labels given
                    self.heading = heading_affix
                elif x_lbl != '':
                    # x label given but not y
                    self.heading = "plot of " + x_lbl
                elif y_lbl != '':
                    # y label given but not x
                    self.heading = y_lbl + " plot"
                else:
                    # neither x nor y given
                    self.title = ''
        # capitalising title
        self.heading = self.heading.title()
        # setting save path
        if path != '' and path != None:
            self.path = path
        else:
            # save path not given
            if self.heading != '' and self.heading != None:
                # heading was constructed earlier
                self.path = self.heading.replace(' ', '_')
            else:
                self.heading = None
