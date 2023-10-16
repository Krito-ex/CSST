def set_template(args):
    # Set the templates here

    if args.template.find('CSST') >= 0:
        args.scheduler = 'CosineAnnealingLR'
        args.max_epoch = 300
        args.learning_rate = 4e-4
