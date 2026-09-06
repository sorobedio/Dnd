import unittest
import argparse
import contextlib
import io
import tempfile
from pathlib import Path

import torch

from workspace.code_generator.model import PrefixCodeGPT
from workspace.code_generator.run import checkpoint_step, default_step, train


class CodeGeneratorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = PrefixCodeGPT(codebook_size=8,references=3,code_length=5,prompt_dim=4,
                                   num_prompts=2,width=16,layers=2,heads=2,step_scale=300).eval()

    def test_step_parsing_and_defaults(self):
        self.assertEqual(checkpoint_step('/ARC-c/250.safetensors'),250)
        self.assertEqual(checkpoint_step('checkpoint-300.pt'),300)
        with self.assertRaises(ValueError):checkpoint_step('last.pt')
        latest={'ARC-c':250,'BoolQ':300}
        self.assertEqual(default_step('ARC-c',latest),(250,'latest_checkpoint_for_task'))
        self.assertEqual(default_step('unseen',latest),(300,'latest_checkpoint_in_training_manifest'))
        self.assertEqual(default_step('ARC-c',latest,100),(100,'explicit'))

    def test_no_future_leakage_and_cache_equivalence(self):
        x=torch.randn(1,7,16)
        full,_=self.model.hidden(x)
        altered=x.clone();altered[:,4:]+=torch.randn_like(altered[:,4:])*10
        changed,_=self.model.hidden(altered)
        torch.testing.assert_close(full[:,:4],changed[:,:4])
        first,cache=self.model.hidden(x[:,:3],cache=True)
        states=[first]
        for i in range(3,7):
            h,cache=self.model.hidden(x[:,i:i+1],past=cache,offset=i,cache=True)
            states.append(h)
        torch.testing.assert_close(full,torch.cat(states,dim=1),atol=1e-6,rtol=1e-5)

    def test_training_and_generation(self):
        prompts=torch.randn(2,2,4);steps=torch.tensor([100,250])
        target=torch.tensor([[1,2,3,4,5,6],[2,6,5,4,3,2]])
        loss,metrics=self.model(prompts,steps,target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(self.model.step[0].weight.grad.abs().sum(),0)
        self.assertGreater(self.model.prompt.weight.grad.abs().sum(),0)
        generated=self.model.generate(prompts,steps)
        self.assertEqual(generated.shape,(2,6))
        self.assertTrue(((generated[:,0]>=0)&(generated[:,0]<3)).all())
        self.assertTrue(((generated[:,1:]>=0)&(generated[:,1:]<8)).all())
        bad=target.clone();bad[0,0]=3
        with self.assertRaises(ValueError):self.model(prompts,steps,bad)

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            data=dict(metadata=dict(num_prompts=2,entries=[dict(dataset='task',checkpoint_step=i+1) for i in range(4)],
                                    latest_steps={'task':4},codebook_size=8,references=3,code_length=5),
                      prompts={'task':torch.randn(5,4)},codes=torch.tensor([[i%3,1,2,3,4,5] for i in range(4)]),
                      schema={},token_shape=[1])
            torch.save(data,root/'data.pt')
            args=argparse.Namespace(dataset=str(root/'data.pt'),output_dir=str(root/'full'),resume=None,device='cpu',
                                    steps=4,batch_size=2,learning_rate=.001,seed=11,eval_every=2,save_every=2,
                                    width=16,layers=2,heads=2,wandb=False,stop_after=0)
            with contextlib.redirect_stdout(io.StringIO()):
                train(args)
                args.output_dir=str(root/'resumed');args.stop_after=2;train(args)
                args.resume=str(root/'resumed/last.pt');args.stop_after=0;train(args)
            full=torch.load(root/'full/last.pt',weights_only=True)
            resumed=torch.load(root/'resumed/last.pt',weights_only=True)
            for key in full['model']:
                torch.testing.assert_close(full['model'][key],resumed['model'][key],rtol=0,atol=0)
            self.assertEqual(full['python_rng'],resumed['python_rng'])
            self.assertEqual(full['best_train_loss'],resumed['best_train_loss'])


if __name__=='__main__':unittest.main()
