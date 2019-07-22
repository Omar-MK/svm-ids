"""
Author: Omar M Khalil 170018405

A class to create FigureMate objects, which, are used to store
    - x and y axis label strings,
    - title string, and
    - save path string.
"""
from datetime import datatime


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
            path:       save path string. If empty string given (default), the
                        object will auto generate a figure save label.
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
        if x_lbl is not None:
            self.x_lbl = string(x_lbl)
        else:
            self.x_lbl = ''


    def set_y(self, y_lbl):
        if y_lbl is not None:
            self.y_lbl = sting(y_lbl)

        else:
            self.y_lbl = ''


    def set_plot_type(self, plot_type):
        if plot_type is not None:
            self.plot_type = sting(plot_type)
        else:
            self.plot_type = ''


    def set_heading(self, heading):
        heading_affix = " plot of " + y_lbl + " vs " +  x_lbl
        if heading is None:
            # no heading wanted
            self.heading = ''
        elif heading != '':
            # heading given
            self.heading = heading
        else:
            # heading not given (empty '') - auto reconstructing
            if self.plot_type != '':
                # plot_type given
                if self.x_lbl != '' and self.y_lbl != '':
                    # both x and y labels given
                    self.heading = self.plot_type + heading_affix
                elif self.x_lbl != '':
                    # x label given but not y
                    self.heading = self.plot_type + " plot of " + self.x_lbl
                else:
                    # y label given but not x, or neither x nor y given
                    self.heading = self.plot_type + " plot"
            else:
                # plot_type not given
                if self.x_lbl != '' and self.y_lbl != '':
                    # both x and y labels given
                    self.heading = heading_affix
                elif self.x_lbl != '':
                    # x label given but not y
                    self.heading = "plot of " + self.x_lbl
                elif self.y_lbl != '':
                    # y label given but not x
                    self.heading = self.y_lbl + " plot"
                else:
                    # neither x nor y given
                    self.title = ''
        # capitalising heading
        self.heading = self.heading.title()

    def set_path(self, path):
        if path != '' and path is not None:
            self.path = string(path) + self.heading 
        else:
            # save path not given
            if self.heading != '' and self.heading is not None:
                # heading was constructed earlier
                self.path = self.heading + datetime.now()
            else:
                self.path = datetime.now()
        self.path = self.path.replace(':', '').replace('.', '').replace(' ', '_')

    def setup_params(self, x_lbl, y_lbl, heading, plot_type, path):
        self.set_plot_type(plot_type)
        self.set_x(x_lbl)
        self.set_y(y_lbl)
        self.set_heading(heading)
        self.set_path(path)
