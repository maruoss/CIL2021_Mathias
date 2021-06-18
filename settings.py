default_settings = {'foo': True, 'bar': False}
my_settings = {'foo': False}
current_settings = default_settings.copy()
current_settings.update(my_settings)