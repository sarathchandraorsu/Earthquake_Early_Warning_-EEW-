# Generated by Django 2.0.5 on 2019-04-29 04:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Remote_User', '0003_clientposts_model_usefulcounts'),
    ]

    operations = [
        migrations.AddField(
            model_name='clientposts_model',
            name='uses',
            field=models.CharField(default='', max_length=100),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='clientposts_model',
            name='tname',
            field=models.CharField(default='', max_length=50),
            preserve_default=False,
        ),
    ]
