import openmdao.api as om


class Paraboloid(om.ExplicitComponent):
    """
    
    """

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)
        self.add_output('h', val = 0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='exact')

    def compute(self, inputs, outputs):
        """
        f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3

        Minimum at: x = 6.6667; y = -7.3333
        """
        x = inputs['x']
        y = inputs['y']

        outputs['f_xy'] = x**3 + 2*y**2 - 4*x - 2*x*y**2
        outputs['h'] = x + y - 1

    def compute_partials(self, inputs, J):
        x = inputs['x']
        y = inputs['y']
        
        J['f_xy', 'x'] = 3*x**2 - 4 - 2*y**2
        J['f_xy', 'y'] = 4*y - 4*x*y
        J['h', 'x'] = 1
        J['h', 'y'] = 1
        


if __name__ == "__main__":

    model = om.Group()
    model.add_subsystem('parab_comp', Paraboloid())

    prob = om.Problem(model)
    prob.setup()

    prob.set_val('parab_comp.x', 3.0)
    prob.set_val('parab_comp.y', -4.0)

    prob.run_model()
    print(prob['parab_comp.f_xy'])

    prob.set_val('parab_comp.x', 5.0)
    prob.set_val('parab_comp.y', -2.0)

    prob.run_model()
    print(prob.get_val('parab_comp.f_xy'))