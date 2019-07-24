"""
Author: Omar M Khalil 170018405

A class to create FigureMate objects, which, are used to store
    - x and y axis label strings,
    - title string, and
    - save path string.
"""
from datetime import datetime


class FigureMate:
    """
    A class to create objects which stores:
        1. strings containing the title, x-axis label, y-axis label prefixes
        2. lists containing tick and legend labels
        3. save path string.
    """

    def __init__(self,
        x_lbl='',
        y_lbl='',
        heading='',
        tick_labels=None,
        legend_labels=None,
        plot_type='',
        prefix=1,
        path=''):
        """
        init method - constructs a FigureMate object.
        Parameters:
            x_lbl:          x axis label - pass None or '' if x label not
                            required.
            y_lbl:          y axis label - pass None or ''  if y label not
                            required.
            heading:        Figure heading label - pass None or '' for empty
                            heading.
            tick_labels:    A list of x-axis tick labels
            legend_labels:  A list of legend labels
            plot_type:      visualisation type or semantics. Used by the object
                            in cases were no heading is provided to auto
                            construct a heading. Note: passing this will
                            trigger heading auto
                        generation.
            path:       save path string. If empty string given (default), the
                        object will auto generate a figure save label.
            prefix:     Boolean to indicate whether the labels in the object
                        are prefixes or full labels
        """
        self.set_x(x_lbl)
        self.set_y(y_lbl)
        self.set_plot_type(plot_type)
        if heading != '' and heading is not None:
            self.set_heading(heading)
        self.set_tick_labels(tick_labels)
        self.set_legend_labels(legend_labels)
        self.set_path(path)
        self.prefix = prefix


    def set_x(self, x_lbl):
        self.x_lbl = self.get_checked_str(x_lbl)


    def set_y(self, y_lbl):
        self.y_lbl = self.get_checked_str(y_lbl)


    def set_heading(self, heading):
        self.heading = self.get_checked_str(heading).title()


    def set_tick_labels(self, tick_labels):
        if isinstance(tick_labels, list) or tick_labels is None:
            self.tick_labels = tick_labels


    def set_legend_labels(self, legend_labels):
        if isinstance(legend_labels, list) or legend_labels is None:
            self.legend_labels = legend_labels


    def set_path(self, path):
        self.path = self.get_checked_str(path) + str(datetime.now())
        self.path = self.path.replace(':', '').replace(' ', '_')
        self.path = self.path[0:2] + self.path[2:].replace('.', '')


    def set_plot_type(self, plot_type):
        self.plot_type = self.get_checked_str(plot_type)
        if self.plot_type != '':
            self.generate_heading()


    def generate_heading(self):
        heading_affix = " plot of " + self.y_lbl + " vs " +  self.x_lbl
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
        self.heading.title()


    def get_checked_str(self, input):
        if input is not None:
            return str(input)
        else:
            return ''
